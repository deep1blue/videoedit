"""Audio processing utilities for AI Video Fixer."""
from __future__ import annotations

import subprocess
from pathlib import Path

import ffmpeg


def run_ffmpeg(cmd: list[str]) -> None:
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.decode('utf-8', 'ignore')}")


def denoise(input_video: Path, output_video: Path, model: str = "rnnoise") -> None:
    """Denoise audio using RNNoise or Demucs."""
    audio_path = input_video.with_suffix(".wav")
    run_ffmpeg(["ffmpeg", "-y", "-i", str(input_video), str(audio_path)])

    if model == "rnnoise":
        import rnnoise_wrapper

        denoised_path = audio_path.with_name("denoised.wav")

        if hasattr(rnnoise_wrapper, "denoise"):
            # old API exposed a convenience function
            rnnoise_wrapper.denoise(str(audio_path), str(denoised_path))
        elif hasattr(rnnoise_wrapper, "RNNoise"):
            # newer versions use an RNNoise class with a filter method
            rn = rnnoise_wrapper.RNNoise()
            if hasattr(rn, "filter"):
                rn.filter(str(audio_path), str(denoised_path))
            elif hasattr(rn, "process"):
                rn.process(str(audio_path), str(denoised_path))
            else:
                # try calling the instance directly
                try:
                    rn(str(audio_path), str(denoised_path))
                except Exception as exc:  # pragma: no cover - extremely unlikely
                    raise AttributeError(
                        "rnnoise_wrapper.RNNoise has no usable filter method"
                    ) from exc
        else:
            raise AttributeError(
                "rnnoise_wrapper does not provide a denoise interface"
            )
    else:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        mdl = get_model("htdemucs_light")
        denoised_path = audio_path.with_name("denoised.wav")
        apply_model(mdl, str(audio_path), str(denoised_path))

    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-i",
        str(denoised_path),
        "-c:v",
        "copy",
        "-map",
        "0:v",
        "-map",
        "1:a",
        str(output_video),
    ])
    audio_path.unlink(missing_ok=True)
    denoised_path.unlink(missing_ok=True)


def normalize_gain(input_video: Path, output_video: Path) -> None:
    """Normalize loudness to -14 LUFS."""
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-af",
        "loudnorm=I=-14:LRA=11:TP=-2",
        str(output_video),
    ])


def cut_fillers(input_video: Path, output_video: Path, filler_ranges: list[tuple[float, float]]) -> None:
    """Remove filler word segments using ffmpeg concat demuxer."""
    if not filler_ranges:
        run_ffmpeg(["ffmpeg", "-y", "-i", str(input_video), "-c", "copy", str(output_video)])
        return

    clips = []
    for i, (start, end) in enumerate(filler_ranges):
        clip_path = input_video.with_name(f"clip_{i}.mp4")
        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c",
            "copy",
            str(clip_path),
        ])
        clips.append(clip_path)

    list_file = input_video.with_name("clips.txt")
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_video),
    ])

    for clip in clips:
        clip.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
