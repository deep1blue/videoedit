"""Video question-answering using SmolVLM2."""
from __future__ import annotations

from pathlib import Path


from transformers import pipeline
import os

# Hugging Face authentication token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_NAME = "apple/smolvlm2"

# Lazy global so model loads only once
_video_qa = None

def _get_pipeline():
    global _video_qa
    if _video_qa is None:
        _video_qa = pipeline(
            "video-question-answering",
            model=MODEL_NAME,
            token=HF_TOKEN,
        )
    return _video_qa


def ask_video_question(video: Path, question: str) -> str:
    """Return the answer to `question` about `video`."""
    qa = _get_pipeline()
    result = qa(video_path=str(video), question=question)
    # pipeline returns dict with 'answer'
    return result.get("answer", "")
