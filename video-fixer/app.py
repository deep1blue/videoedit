"""Gradio interface for AI Video Fixer."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import gradio as gr

from processing.video_utils import (
    stabilize,
    focus_speaker,
    blur_bg,
    adjust_levels,
)
from processing.audio_utils import denoise, normalize_gain, cut_fillers
from processing.transcript import get_transcript, find_filler_words

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


class VideoState:
    """Simple state container holding the current video path."""

    def __init__(self, path: Path | None = None):
        self.path = path


state = VideoState()


def _new_temp_path(suffix: str = ".mp4") -> Path:
    fd, temp = tempfile.mkstemp(dir=TMP_DIR, suffix=suffix)
    Path(temp).unlink()  # only need the name
    return Path(temp)


def upload(video: Path) -> Path:
    dest = TMP_DIR / "input.mp4"
    shutil.copy(video, dest)
    state.path = dest
    return dest


def apply_video_effect(func) -> Path:
    if not state.path:
        raise gr.Error("No video uploaded")
    output = _new_temp_path()
    func(state.path, output)
    state.path = output
    return output


def do_stabilize() -> Path:
    return apply_video_effect(stabilize)


def do_focus() -> Path:
    return apply_video_effect(focus_speaker)


def do_blur() -> Path:
    return apply_video_effect(blur_bg)


def do_contrast() -> Path:
    return apply_video_effect(adjust_levels)


def do_denoise() -> Path:
    return apply_video_effect(denoise)


def do_normalize() -> Path:
    return apply_video_effect(normalize_gain)


def do_remove_fillers() -> Path:
    if not state.path:
        raise gr.Error("No video uploaded")
    # extract audio
    audio_path = _new_temp_path(suffix=".wav")
    cut_video = _new_temp_path()
    # extract audio for transcription
    subprocess_cmd = ["ffmpeg", "-y", "-i", str(state.path), str(audio_path)]
    subprocess.run(subprocess_cmd, check=True)
    transcript = get_transcript(audio_path)
    ranges = find_filler_words(transcript)
    cut_fillers(state.path, cut_video, ranges)
    state.path = cut_video
    return cut_video


def cleanup() -> None:
    shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir()
    state.path = None


with gr.Blocks() as demo:
    gr.Markdown("# AI Video Fixer")
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input Video")
            stabilize_btn = gr.Button("Stabilise")
            focus_btn = gr.Button("Focus on Speaker")
            blur_btn = gr.Button("Blur Background")
            contrast_btn = gr.Button("Auto Contrast/Colour")
            denoise_btn = gr.Button("Denoise Audio")
            normalize_btn = gr.Button("Normalise Volume")
            filler_btn = gr.Button("Remove Filler Words")
            reset_btn = gr.Button("Reset")
        with gr.Column():
            video_out = gr.Video(label="Preview")

    video_in.upload(upload, inputs=video_in, outputs=video_out)
    stabilize_btn.click(do_stabilize, outputs=video_out)
    focus_btn.click(do_focus, outputs=video_out)
    blur_btn.click(do_blur, outputs=video_out)
    contrast_btn.click(do_contrast, outputs=video_out)
    denoise_btn.click(do_denoise, outputs=video_out)
    normalize_btn.click(do_normalize, outputs=video_out)
    filler_btn.click(do_remove_fillers, outputs=video_out)
    reset_btn.click(cleanup)

if __name__ == "__main__":
    demo.launch()
