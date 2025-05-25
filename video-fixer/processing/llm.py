"""Video question-answering using the lightweight SmolVLM2-256M model."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModel
import cv2
from typing import List

# Pull your HF token from the environment
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("Please set the HF_TOKEN environment variable before running.")

# The smallest SmolVLM2 variant with video-QA support
MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# Global singletons so we only load once
_processor: AutoProcessor | None = None
_model: torch.nn.Module | None = None

# Pick the best device available
if torch.cuda.is_available():
    _device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"


def _get_model_and_processor() -> tuple[AutoProcessor, torch.nn.Module]:
    global _processor, _model
    if _processor is None or _model is None:
        # Use trust_remote_code so we can pull in custom classes
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        _model.to(_device)
    return _processor, _model


def _sample_video_frames(path: Path, num_frames: int = 8) -> List["np.ndarray"]:
    """Load and uniformly sample up to ``num_frames`` RGB frames from a video."""
    import numpy as np  # opencv requires numpy but delay import for tests

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(total // num_frames, 1)
    frames: List[np.ndarray] = []
    frame_idx = 0
    while len(frames) < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += step
    cap.release()
    return frames


def ask_video_question(video: Path, question: str) -> str:
    """
    Ask a natural-language question about a video file and return the answer.

    Args:
        video: Path to your local video file (e.g. 'tmp/input.mp4').
        question: A string question (e.g. "What color is the car?").

    Returns:
        The model’s text answer.
    """
    processor, model = _get_model_and_processor()

    frames = _sample_video_frames(video)

    # Pass frames as images since the processor may not accept a raw video path
    inputs = processor(images=frames, text=question, return_tensors="pt").to(_device)

    # Next, we check if the model has a `.generate()` method:
    if not hasattr(model, "generate"):
        raise AttributeError(
            "Loaded model does not have a `.generate()` method. "
            "Please confirm the correct model class supports text generation."
        )

    # If `.generate()` exists, proceed
    outputs = model.generate(**inputs)
    # Or if there's a different method, e.g. model.predict(...)
    
    # decode the results
    if hasattr(processor, "decode"):
        return processor.decode(outputs[0], skip_special_tokens=True)
    else:
        # If the custom code expects something else:
        return outputs[0].cpu().numpy().tolist()
