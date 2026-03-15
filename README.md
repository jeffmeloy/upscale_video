```
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

```