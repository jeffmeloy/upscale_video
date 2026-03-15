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
                        Default: <input>_<res>p[_upscale-<level>][_denoise-<level>][_deblur-<level>]_fps<fps>_cq<cq>.mp4

  --res RES             Target min pixel dimension (default: 2160). Cannot be used with --scale.
                        Auto-computes scale factor (2x, 3x, or 4x).

  --scale {1,2,3,4}     Explicit scale factor. Cannot be used with --res.
                        1 = same resolution
                        2 = 2x   3 = 3x   4 = 4x

  --upscale PRESET      BICUBIC, LOW, MEDIUM, HIGH, ULTRA

  --denoise PRESET      LOW, MEDIUM, HIGH, ULTRA

  --deblur PRESET       LOW, MEDIUM, HIGH, ULTRA

  --double_fps          Double the source frame rate via ffmpeg minterpolate blend.

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
# 4K ULTRA CQ20
  python upscale_video.py input.mp4

# 4x, near-lossless                          
  python upscale_video.py input.mp4 --scale 4 --cq 18

# Denoise + 4K upscale
  python upscale_video.py input.mp4 --denoise HIGH

# Denoise + deblur + interpolate frames, no upscale, near-lossless
  python upscale_video.py input.mp4 --deblur ULTRA --denoise ULTRA --double_fps --cq 18

# Denoise only, no upscale
  python upscale_video.py input.mp4 --scale 1 --denoise ULTRA
```