"""Video question-answering using the lightweight SmolVLM2-256M model."""
from __future__ import annotations

import os
from pathlib import Path

from transformers import pipeline
import cv2
from PIL import Image

# Hugging Face authentication token from environment
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Smallest SmolVLM2 variant for video QA
MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# Lazy-global so we only load the model once
_video_qa = None

def _get_pipeline():
    global _video_qa
    if _video_qa is None:
        # Use the "visual-question-answering" task name (pipeline registry)
        _video_qa = pipeline(
            "visual-question-answering",
            model=MODEL_NAME,
            use_auth_token=HF_TOKEN,  # correct auth flag
            trust_remote_code=True,    # allows any custom pipeline code in the repo
        )
    return _video_qa

def ask_video_question(video: Path, question: str) -> str:
    """
    Ask a natural-language question about a video file and return the answer.

    Args:
        video:    Path to the local video file.
        question: The question you want answered (e.g. "What color is the car?").

    Returns:
        The model’s answer as plain text.
    """
    qa = _get_pipeline()

    # Extract the first frame to use with image-based VQA pipelines
    cap = cv2.VideoCapture(str(video))
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Failed to read video: {video}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    # Fallback to standard image-based pipeline if the custom pipeline isn't available
    result = qa(image=image, question=question)
    return result.get("answer", "")
