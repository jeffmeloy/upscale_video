#!/usr/bin/env python3
"""
Video upscaler using NVIDIA RTX Video Super Resolution (nvidia-vfx).

Requirements:
  pip install nvidia-vfx av numpy tqdm

  pip install torch --index-url https://download.pytorch.org/whl/cu124
  (or cu121, cu118 depending on your CUDA toolkit version)

  ffmpeg on PATH (--double_fps: minterpolate blend)

  GPU: NVIDIA RTX (Turing, Ampere, Ada, Blackwell, or Hopper)

Arguments:
  input                 Input video file path (positional, required).
  -o, --output          Output file path.
                        Default: <input>_<res>p[_<upscale>][_denoise-<level>][_deblur-<level>]_fps<fps>_cq<cq>.mp4

  --res RES             Target min dimension in pixels (default: 2160 for 4K).
                        Auto-computes scale factor (2x, 3x, or 4x).
                        Cannot be used with --scale.

  --scale {1,2,3,4}     Explicit scale factor. Cannot be used with --res.
                        1 = same resolution (preprocess only)
                        2 = 2x   3 = 3x   4 = 4x

  --upscale PRESET      RTX VSR upscale quality (default: ULTRA).
                        BICUBIC, LOW, MEDIUM, HIGH, ULTRA

  --denoise PRESET      Denoise before upscaling (reduces grain/noise).
                        LOW, MEDIUM, HIGH, ULTRA

  --deblur PRESET       Deblur before upscaling (sharpens soft/blurry footage).
                        LOW, MEDIUM, HIGH, ULTRA

  --double_fps          Double the source frame rate via ffmpeg minterpolate blend.
                        Streamed through raw pipes (no temp file, no re-encode).

  --pad                 Pad to mod-8 alignment instead of cropping (default: crop).
                        Preserves all original pixels; adds black border.

  --cq CQ               Constant quality level 0-51 (default: 20).
                        0 = lossless, 18 = visually lossless, 22 = good, 28+ = small files.
                        Encoder auto-allocates bitrate per scene complexity.

Pipeline order (fixed, stages skipped if not requested):
  denoise → deblur → interpolate → upscale

Encoder:
  Uses hevc_nvenc (GPU NVENC) in VBR/CQ mode with 80 Mbps ceiling.
  Output dims rounded to mod-8. Audio: stream copy if MP4-compatible, else AAC at source bitrate.

Examples:
  python upscale_video.py input.mp4                                # 4K ULTRA CQ20
  python upscale_video.py input.mp4 --scale 4 --cq 18             # 4x, near-lossless
  python upscale_video.py input.mp4 --denoise HIGH                 # denoise + 4K upscale
  python upscale_video.py input.mp4 --deblur ULTRA --denoise ULTRA --double_fps --cq 18
  python upscale_video.py input.mp4 --scale 1 --denoise ULTRA     # denoise only, no upscale
"""

import argparse
import math
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
import torch
import torch.nn.functional as F
from nvvfx import VideoSuperRes
from tqdm import tqdm

DEV = "cuda:0"
HEVC_MAX = 8192

_CS_MATRICES = {
    "bt601": [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    "bt709": [
        [1.0, 0.0, 1.5748],
        [1.0, -0.1873, -0.4681],
        [1.0, 1.8556, 0.0],
    ],
    "bt2020": [
        [1.0, 0.0, 1.4746],
        [1.0, -0.1646, -0.5714],
        [1.0, 1.8814, 0.0],
    ],
}

_AVCOL_SPC_BT709 = 1
_AVCOL_SPC_BT470BG = 5
_AVCOL_SPC_SMPTE170M = 6
_AVCOL_SPC_BT2020_NCL = 9
_AVCOL_SPC_BT2020_CL = 10

_UPSCALE_QUALITIES = ["BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA"]
_PREPROCESS_LEVELS = ["LOW", "MEDIUM", "HIGH", "ULTRA"]


def detect_colorspace(vs):
    try:
        cs_val = int(vs.codec_context.colorspace)
    except Exception:
        cs_val = 0
    if cs_val == _AVCOL_SPC_BT709:
        return "bt709"
    if cs_val in (_AVCOL_SPC_BT2020_NCL, _AVCOL_SPC_BT2020_CL):
        return "bt2020"
    if cs_val in (_AVCOL_SPC_BT470BG, _AVCOL_SPC_SMPTE170M):
        return "bt601"
    w, h = vs.codec_context.width, vs.codec_context.height
    return "bt709" if min(w, h) >= 720 else "bt601"


def detect_full_range(vs):
    """Detect whether source uses full (0-255) or limited (16-235) range."""
    # yuvj* pixel formats are always full range
    if vs.codec_context.pix_fmt in ("yuvj420p", "yuvj422p", "yuvj444p"):
        return True
    # Check explicit color_range metadata (2 = full/JPEG, 1 = limited/MPEG)
    try:
        cr = int(vs.codec_context.color_range)
        if cr == 2:
            return True
        if cr == 1:
            return False
    except Exception:
        pass
    # Default: limited range (vast majority of video content)
    return False


def build_yuv2rgb(name):
    return torch.tensor(_CS_MATRICES[name], dtype=torch.float32, device=DEV)


def compute_scale(w, h, target):
    return max(2, min(4, math.ceil(target / min(w, h)))) if min(w, h) > 0 else 4


def align_dims(w, h, pad=False):
    """Round dimensions to mod-8 for nvvfx/NVENC.

    Default (crop): rounds down, losing up to 7 pixels per axis.
    With pad: rounds up, adding black border pixels.
    """
    if pad:
        w = math.ceil(w / 8) * 8
        h = math.ceil(h / 8) * 8
    else:
        w = (w // 8) * 8
        h = (h // 8) * 8
    return w, h


def probe(path):
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        ctx = vs.codec_context
        return (
            ctx.width,
            ctx.height,
            float(vs.average_rate or 30),
            vs.frames or 0,
            len(c.streams.audio) > 0,
            ctx.pix_fmt in ("yuv420p", "yuvj420p"),
        )


_CS_PARAMS = {
    "bt601": {"colorspace": 5, "color_primaries": 5, "color_trc": 6},
    "bt709": {"colorspace": 1, "color_primaries": 1, "color_trc": 1},
    "bt2020": {"colorspace": 9, "color_primaries": 9, "color_trc": 14},
}


def open_encoder(path, w, h, fps, cq, cs_name="bt709", full_range=False):
    rate = Fraction(fps).limit_denominator(10000)
    cont = av.open(str(path), mode="w")
    s = cont.add_stream("hevc_nvenc", rate=rate)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    cs = _CS_PARAMS.get(cs_name, _CS_PARAMS["bt709"])
    s.codec_context.colorspace = cs["colorspace"]
    s.codec_context.color_primaries = cs["color_primaries"]
    s.codec_context.color_trc = cs["color_trc"]
    s.codec_context.color_range = 2 if full_range else 1  # JPEG=full, MPEG=limited
    s.codec_context.options = {
        "rc": "vbr",
        "cq": str(cq),
        "b:v": "0",
        "maxrate": "80M",
        "bufsize": "160M",
    }
    s.codec_context.open()
    return cont, s


def load_model(label, quality_str, in_w, in_h, out_w, out_h, gpu=0):
    """Load a VideoSuperRes model and return it."""
    print(
        f"  {label}: {quality_str} ({in_w}x{in_h} -> {out_w}x{out_h})...",
        end=" ",
        flush=True,
    )
    sr = VideoSuperRes(device=gpu, quality=VideoSuperRes.QualityLevel[quality_str])
    sr.input_width, sr.input_height = in_w, in_h
    sr.output_width, sr.output_height = out_w, out_h
    sr.load()
    print("done.")
    return sr


class MinterpolateStream:
    """Streaming frame interpolation via ffmpeg minterpolate over raw pipe"""

    def __init__(self, w, h, src_fps, dst_fps):
        self.w, self.h = w, h
        self.frame_bytes = w * h * 3

        minterp = f"minterpolate=fps={dst_fps}:mi_mode=blend"
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(src_fps),
                "-i",
                "pipe:0",
                "-vf",
                minterp,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=self.frame_bytes * 4,
        )
        self._write_error = None

    def _feed_thread(self, frame_iter):
        try:
            for rgb in frame_iter:
                self.proc.stdin.write(rgb.tobytes())
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        except Exception as e:
            self._write_error = e
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def stream(self, frame_iter):
        writer = threading.Thread(target=self._feed_thread, args=(frame_iter,))
        writer.start()
        buf = b""
        try:
            while True:
                needed = self.frame_bytes - len(buf)
                if needed > 0:
                    chunk = self.proc.stdout.read(needed)
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) < self.frame_bytes:
                        continue
                yield (
                    np.frombuffer(buf[: self.frame_bytes], dtype=np.uint8)
                    .reshape(self.h, self.w, 3)
                    .copy()
                )
                buf = buf[self.frame_bytes :]
        finally:
            writer.join()
            self.proc.stdout.close()
            self.proc.wait()
        if self._write_error:
            raise self._write_error

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


class FrameIO:
    """Pre-allocated, zero-malloc-per-frame GPU converter."""

    def __init__(
        self,
        h,
        w,
        out_h,
        out_w,
        yuv2rgb,
        has_preprocess=False,
        src_h=None,
        src_w=None,
        full_range=False,
    ):
        self.h, self.w = h, w
        # Source dimensions (smaller than h,w when padding, equal when cropping)
        self.src_h = src_h or h
        self.src_w = src_w or w
        self.full_range = full_range
        self.yuv2rgb = yuv2rgb
        self.y_pin = torch.zeros((h, w), dtype=torch.uint8, pin_memory=True)
        self.uv_pin = torch.zeros(
            (2, h // 2, w // 2), dtype=torch.uint8, pin_memory=True
        )
        self.rgb_pin = torch.zeros((h, w, 3), dtype=torch.uint8, pin_memory=True)
        self.y_gpu = torch.zeros((h, w), dtype=torch.uint8, device=DEV)
        self.uv_gpu_u8 = torch.zeros((2, h // 2, w // 2), dtype=torch.uint8, device=DEV)
        self.uv_gpu = torch.zeros(
            (1, 2, h // 2, w // 2), dtype=torch.float32, device=DEV
        )
        self.yuv_chw = torch.zeros((3, h * w), dtype=torch.float32, device=DEV)
        self.input_chw = torch.zeros((3, h, w), dtype=torch.float32, device=DEV)
        self.out_f32 = torch.empty((3, out_h, out_w), dtype=torch.float32, device=DEV)
        self.out_hwc = torch.empty((out_h, out_w, 3), dtype=torch.uint8, device=DEV)
        self.out_pin = torch.empty(
            (out_h, out_w, 3), dtype=torch.uint8, pin_memory=True
        )
        if has_preprocess:
            self.pre_f32 = torch.empty((3, h, w), dtype=torch.float32, device=DEV)
            self.pre_hwc = torch.empty((h, w, 3), dtype=torch.uint8, device=DEV)
            self.pre_pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

    @staticmethod
    def _plane_to_pin(buf, plane, rows, cols):
        raw = np.frombuffer(plane, np.uint8).reshape(rows, plane.line_size)[
            :rows, :cols
        ]
        buf[:rows, :cols].copy_(torch.from_numpy(np.ascontiguousarray(raw)))

    def yuv_to_input(self, frame):
        h, w = self.h, self.w
        sh, sw = self.src_h, self.src_w
        self._plane_to_pin(self.y_pin, frame.planes[0], sh, sw)
        self._plane_to_pin(self.uv_pin[0], frame.planes[1], sh // 2, sw // 2)
        self._plane_to_pin(self.uv_pin[1], frame.planes[2], sh // 2, sw // 2)
        self.y_gpu.copy_(self.y_pin, non_blocking=True)
        self.uv_gpu_u8.copy_(self.uv_pin, non_blocking=True)
        self.uv_gpu[0].copy_(self.uv_gpu_u8)
        uv_up = F.interpolate(self.uv_gpu, (h, w), mode="bilinear", align_corners=False)
        if self.full_range:
            self.yuv_chw[0].copy_(self.y_gpu.view(-1))
            self.yuv_chw[1].copy_(uv_up[0, 0].view(-1)).sub_(128.0)
            self.yuv_chw[2].copy_(uv_up[0, 1].view(-1)).sub_(128.0)
        else:
            # Limited range: Y [16,235] → [0,255], UV [16,240] → centered & scaled
            self.yuv_chw[0].copy_(self.y_gpu.view(-1)).sub_(16.0).mul_(255.0 / 219.0)
            self.yuv_chw[1].copy_(uv_up[0, 0].view(-1)).sub_(128.0).mul_(255.0 / 224.0)
            self.yuv_chw[2].copy_(uv_up[0, 1].view(-1)).sub_(128.0).mul_(255.0 / 224.0)
        torch.mm(self.yuv2rgb, self.yuv_chw, out=self.input_chw.view(3, -1))
        self.input_chw.div_(255.0)
        return self.input_chw

    def rgb_to_input(self, frame):
        sh, sw = self.src_h, self.src_w
        self.rgb_pin[:sh, :sw].copy_(
            torch.from_numpy(frame.to_ndarray(format="rgb24")[:sh, :sw])
        )
        self.input_chw.copy_(
            self.rgb_pin.to(DEV, non_blocking=True).permute(2, 0, 1)
        ).div_(255.0)
        return self.input_chw

    def np_rgb_to_input(self, rgb_np):
        self.rgb_pin.copy_(torch.from_numpy(rgb_np))
        self.input_chw.copy_(
            self.rgb_pin.to(DEV, non_blocking=True).permute(2, 0, 1)
        ).div_(255.0)
        return self.input_chw

    def run_preprocess(self, chw, pre_models, stream_ptr):
        """Chain preprocess models on GPU. Returns final CHW tensor."""
        for sr in pre_models:
            result = sr.run(chw, stream_ptr=stream_ptr)
            self.pre_f32.copy_(torch.from_dlpack(result.image))
            chw = self.pre_f32
        return chw

    def preprocess_to_numpy(self, chw):
        """(3,H,W) float [0,1] GPU → (H,W,3) uint8 numpy for minterpolate."""
        self.pre_hwc.copy_(
            chw.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).permute(1, 2, 0)
        )
        self.pre_pin.copy_(self.pre_hwc)
        torch.cuda.current_stream().synchronize()
        return self.pre_pin.numpy().copy()

    def chw_to_frame(self, chw):
        """(3,H,W) float [0,1] GPU tensor → av.VideoFrame (RGB24). For scale=1 output."""
        self.out_hwc.copy_(
            chw.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).permute(1, 2, 0)
        )
        self.out_pin.copy_(self.out_hwc)
        torch.cuda.current_stream().synchronize()
        return av.VideoFrame.from_ndarray(self.out_pin.numpy(), format="rgb24")

    def upscale_to_frame(self, dlpack_result):
        """DLPack CHW float [0,1] → av.VideoFrame (RGB24)."""
        self.out_f32.copy_(torch.from_dlpack(dlpack_result.image))
        self.out_hwc.copy_(self.out_f32.clamp_(0.0, 1.0).mul_(255.0).permute(1, 2, 0))
        self.out_pin.copy_(self.out_hwc)
        torch.cuda.current_stream().synchronize()
        return av.VideoFrame.from_ndarray(self.out_pin.numpy(), format="rgb24")


def main():
    p = argparse.ArgumentParser(
        description="RTX VSR upscaler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input video file.")
    p.add_argument("-o", "--output", default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--res", type=int, default=None, help="Target min dimension.")
    g.add_argument("--scale", type=int, choices=[1, 2, 3, 4], default=None)
    p.add_argument(
        "--upscale",
        default="ULTRA",
        choices=_UPSCALE_QUALITIES,
        help="Upscale quality preset.",
    )
    p.add_argument(
        "--cq",
        type=int,
        default=20,
        help="Constant quality 0-51 (0=lossless, 20=default, lower=better).",
    )
    p.add_argument(
        "--denoise",
        default=None,
        choices=_PREPROCESS_LEVELS,
        metavar="LEVEL",
        help="Denoise before upscaling (LOW, MEDIUM, HIGH, ULTRA).",
    )
    p.add_argument(
        "--deblur",
        default=None,
        choices=_PREPROCESS_LEVELS,
        metavar="LEVEL",
        help="Deblur before upscaling (LOW, MEDIUM, HIGH, ULTRA).",
    )
    p.add_argument(
        "--double_fps",
        action="store_true",
        help="Double frame rate via ffmpeg minterpolate blend.",
    )
    p.add_argument(
        "--pad",
        action="store_true",
        help="Pad to mod-8 alignment instead of cropping (preserves all pixels).",
    )
    args = p.parse_args()

    src = Path(args.input).resolve()
    if not src.exists():
        sys.exit(f"Error: {src} not found")

    in_w, in_h, fps, n_frames, has_audio, is_yuv = probe(src)
    do_interp = args.double_fps
    out_fps = fps * 2 if do_interp else fps
    est_frames = (2 * n_frames - 1) if do_interp and n_frames else n_frames

    scale = (
        args.scale
        if args.scale is not None
        else compute_scale(in_w, in_h, args.res or 2160)
    )
    do_upscale = scale > 1
    align_w, align_h = align_dims(in_w, in_h, pad=args.pad)
    out_w, out_h = (
        (align_w * scale, align_h * scale) if do_upscale else (align_w, align_h)
    )
    if max(out_w, out_h) > HEVC_MAX:
        sys.exit(f"Error: {out_w}x{out_h} exceeds HEVC max {HEVC_MAX}")

    # --- Build default output filename ---
    if args.output:
        dst = Path(args.output)
    else:
        parts = [src.stem]
        parts.append(f"{out_h}p")
        if do_upscale:
            parts.append(f"upscale-{args.upscale.lower()}")
        if args.denoise:
            parts.append(f"denoise-{args.denoise.lower()}")
        if args.deblur:
            parts.append(f"deblur-{args.deblur.lower()}")
        parts.append(f"fps{out_fps:g}")
        parts.append(f"cq{args.cq}")
        dst = src.parent / f"{'_'.join(parts)}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)

    has_preprocess = args.denoise or args.deblur
    do_pad = args.pad and (align_w, align_h) != (in_w, in_h)
    do_crop = not args.pad and (align_w, align_h) != (in_w, in_h)

    if (align_w, align_h) != (in_w, in_h):
        mode = "Pad" if args.pad else "Crop"
        print(f"{mode:10s}: {in_w}x{in_h} -> {align_w}x{align_h} (mod-8 alignment)")

    pipeline_parts = []
    if args.denoise:
        pipeline_parts.append(f"denoise({args.denoise})")
    if args.deblur:
        pipeline_parts.append(f"deblur({args.deblur})")
    if do_interp:
        pipeline_parts.append("interpolate(blend)")
    if do_upscale:
        pipeline_parts.append(f"upscale({args.upscale})")

    print(
        f"Input     : {src}\n"
        f"Resolution: {in_w}x{in_h} -> {out_w}x{out_h}"
        + (f" ({scale}x)" if do_upscale else "")
        + "\n"
        f"FPS       : {fps:.2f}"
        + (f" -> {out_fps:.1f}" if do_interp else "")
        + (f"  Frames: ~{est_frames}" if est_frames else "")
        + "\n"
        f"Pipeline  : {' → '.join(pipeline_parts)}\n"
        f"Encode CQ : {args.cq}" + ("\nAudio     : no" if not has_audio else "")
    )

    torch.cuda.set_device(0)
    stream_ptr = torch.cuda.current_stream().cuda_stream

    print("Loading models...")
    denoise_model = None
    deblur_model = None
    upscale_model = None

    if args.denoise:
        denoise_model = load_model(
            "Denoise", f"DENOISE_{args.denoise}", align_w, align_h, align_w, align_h
        )
    if args.deblur:
        deblur_model = load_model(
            "Deblur", f"DEBLUR_{args.deblur}", align_w, align_h, align_w, align_h
        )
    if do_upscale:
        upscale_model = load_model(
            "Upscale", args.upscale, align_w, align_h, out_w, out_h
        )

    pre_models = [m for m in (denoise_model, deblur_model) if m is not None]
    inp = av.open(str(src))
    vs = inp.streams.video[0]
    vs.thread_type = "AUTO"
    is_yuv = vs.codec_context.pix_fmt in ("yuv420p", "yuvj420p")
    cs_name = detect_colorspace(vs)
    full_range = detect_full_range(vs)
    yuv2rgb = build_yuv2rgb(cs_name)
    print(
        f"Colorspace: {cs_name.upper()} ({'full' if full_range else 'limited'} range)"
    )

    out_cont, vid_stream = open_encoder(
        dst, out_w, out_h, out_fps, args.cq, cs_name, full_range=True
    )
    print("Encoder   : hevc_nvenc VBR/CQ")

    _MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "flac", "opus", "alac"}
    aud_in = inp.streams.audio[0] if has_audio else None
    aud_out = None
    aud_copy = False
    if aud_in:
        codec_name = aud_in.codec_context.name
        if codec_name in _MP4_AUDIO:
            for attempt in [
                lambda: out_cont.add_stream(template=aud_in),
                lambda: out_cont.add_stream(aud_in),
            ]:
                try:
                    aud_out = attempt()
                    aud_copy = True
                    print(f"Audio     : stream copy ({codec_name})")
                    break
                except (TypeError, ValueError, av.error.FFmpegError):
                    continue
        if not aud_copy:
            src_abr = aud_in.codec_context.bit_rate or 128_000
            aud_out = out_cont.add_stream("aac", rate=aud_in.rate)
            aud_out.bit_rate = max(src_abr, 128_000)
            print(
                f"Audio     : re-encode {codec_name} → AAC {aud_out.bit_rate // 1000}kbps"
            )

    fio = FrameIO(
        align_h,
        align_w,
        out_h,
        out_w,
        yuv2rgb,
        has_preprocess=bool(has_preprocess),
        src_h=in_h if do_pad else None,
        src_w=in_w if do_pad else None,
        full_range=full_range,
    )
    print()
    n, t0 = 0, time.time()

    def decode_to_gpu(frame):
        if is_yuv:
            return fio.yuv_to_input(frame)
        return fio.rgb_to_input(frame)

    def encode_frame(vf):
        nonlocal n
        for pkt in vid_stream.encode(vf):
            out_cont.mux(pkt)
        n += 1

    if do_interp:
        pre_bar = None
        if has_preprocess:
            pre_label = "+".join(
                ([f"Denoise"] if denoise_model else [])
                + ([f"Deblur"] if deblur_model else [])
            )
            pre_bar = (
                tqdm(total=n_frames, unit="f", desc=pre_label) if n_frames else None
            )

        def source_frames():
            for pkt in inp.demux(vs):
                for frame in pkt.decode():
                    if not isinstance(frame, av.VideoFrame):
                        continue
                    if pre_models:
                        chw = decode_to_gpu(frame)
                        chw = fio.run_preprocess(chw, pre_models, stream_ptr)
                        result = fio.preprocess_to_numpy(chw)
                    else:
                        rgb = frame.to_ndarray(format="rgb24")
                        if do_pad:
                            padded = np.zeros((align_h, align_w, 3), dtype=np.uint8)
                            padded[:in_h, :in_w] = rgb
                            result = padded
                        else:
                            result = rgb[:align_h, :align_w].copy()
                    if pre_bar:
                        pre_bar.update(1)
                    yield result

        minterp = MinterpolateStream(align_w, align_h, fps, out_fps)
        up_label = "Upscale" if do_upscale else "Encode"
        up_bar = tqdm(total=est_frames, unit="f", desc=up_label) if est_frames else None

        try:
            for rgb in minterp.stream(source_frames()):
                chw = fio.np_rgb_to_input(rgb)
                if upscale_model:
                    vf = fio.upscale_to_frame(
                        upscale_model.run(chw, stream_ptr=stream_ptr)
                    )
                else:
                    vf = fio.chw_to_frame(chw)
                encode_frame(vf)
                if up_bar:
                    up_bar.update(1)
        finally:
            minterp.close()

        if pre_bar:
            pre_bar.close()
        if up_bar:
            up_bar.close()

    else:
        parts = []
        if denoise_model:
            parts.append("Denoise")
        if deblur_model:
            parts.append("Deblur")
        if do_upscale:
            parts.append("Upscale")
        label = "+".join(parts) if parts else "Encode"
        bar = tqdm(total=n_frames, unit="f", desc=label) if n_frames else None

        for pkt in inp.demux(vs):
            for frame in pkt.decode():
                if not isinstance(frame, av.VideoFrame):
                    continue
                chw = decode_to_gpu(frame)
                if pre_models:
                    chw = fio.run_preprocess(chw, pre_models, stream_ptr)
                if upscale_model:
                    vf = fio.upscale_to_frame(
                        upscale_model.run(chw, stream_ptr=stream_ptr)
                    )
                else:
                    vf = fio.chw_to_frame(chw)
                encode_frame(vf)
                if bar:
                    bar.update(1)

        if bar:
            bar.close()

    if aud_in and aud_out:
        inp.seek(0)
        if aud_copy:
            print("Copying audio...", end=" ", flush=True)
            a_n = 0
            for pkt in inp.demux(aud_in):
                if pkt.dts is not None:
                    pkt.stream = aud_out
                    out_cont.mux(pkt)
                    a_n += 1
            print(f"done ({a_n} packets).")
        else:
            print("Encoding audio...", end=" ", flush=True)
            a_n = 0
            for pkt in inp.demux(aud_in):
                for af in pkt.decode():
                    for p in aud_out.encode(af):
                        out_cont.mux(p)
                        a_n += 1
            for p in aud_out.encode(None):
                out_cont.mux(p)
                a_n += 1
            print(f"done ({a_n} packets).")

    for p in vid_stream.encode(None):
        out_cont.mux(p)
    out_cont.close()
    inp.close()
    for m in pre_models:
        m.close()
    if upscale_model:
        upscale_model.close()

    dt = time.time() - t0
    print(f"\nProcessed {n} frames in {dt:.1f}s ({n / dt:.1f} fps)")
    print(f"Output    : {dst} ({dst.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
