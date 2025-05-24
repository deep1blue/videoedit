"""Transcription utilities and filler-word detection."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from faster_whisper import WhisperModel
import nltk


MODEL_CACHE = Path("models")


def get_transcript(audio_path: Path) -> List[Tuple[float, float, str]]:
    """Return a list of (start, end, text) tuples using faster-whisper."""
    model = WhisperModel("tiny-int8", download_root=str(MODEL_CACHE))
    segments, _ = model.transcribe(str(audio_path))
    results = []
    for segment in segments:
        results.append((segment.start, segment.end, segment.text))
    return results


CUSTOM_FILLERS = {"um", "uh", "like"}


def find_filler_words(transcript: List[Tuple[float, float, str]]) -> List[Tuple[float, float]]:
    """Return list of (start, end) ranges containing filler words."""
    nltk.download("stopwords", quiet=True)
    filler_words = set(nltk.corpus.stopwords.words("english")) | CUSTOM_FILLERS
    ranges = []
    for start, end, text in transcript:
        tokens = [t.lower() for t in text.split()]
        if any(token in filler_words for token in tokens):
            ranges.append((start, end))
    return ranges
