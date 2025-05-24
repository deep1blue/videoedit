"""Video question-answering using the lightweight SmolVLM2-256M model."""

from __future__ import annotations

import os
from pathlib import Path

import torch

try:  # transformers < 4.40 may not expose AutoModelForConditionalGeneration
    from transformers import AutoProcessor, AutoModelForConditionalGeneration
except ImportError:  # pragma: no cover - fallback for older versions
    raise ImportError(
        "Transformers >= 4.40 is required to run the SmolVLM model. "
        "Please upgrade the 'transformers' package."
    )

# Pull your HF token from the environment
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("Please set the HF_TOKEN environment variable before running.")

# The smallest SmolVLM2 variant with video-QA support
MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# Global singletons so we only load once
_processor: AutoProcessor | None = None
_model: AutoModelForConditionalGeneration | None = None

# Pick the best device available
if torch.cuda.is_available():
    _device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"


def _get_model_and_processor() -> (
    tuple[AutoProcessor, AutoModelForConditionalGeneration]
):
    global _processor, _model
    if _processor is None or _model is None:
        # Load both with trust_remote_code so we pull in any custom classes
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            use_auth_token=HF_TOKEN,
            trust_remote_code=True,
        )
        _model = AutoModelForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            use_auth_token=HF_TOKEN,
            trust_remote_code=True,
        )
        _model.to(_device)
    return _processor, _model


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

    # Prepare inputs: the processor will load & sample frames internally
    inputs = processor(
        video=str(video),
        text=question,
        return_tensors="pt",
    ).to(_device)

    # Generate an answer
    generated = model.generate(**inputs)
    answer = processor.decode(generated[0], skip_special_tokens=True)
    return answer
