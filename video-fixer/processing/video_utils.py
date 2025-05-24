"""Video processing utilities for AI Video Fixer."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import ffmpeg
from vidstab import VidStab
import mediapipe as mp


def run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command and raise if it fails."""
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.decode('utf-8', 'ignore')}")


def stabilize(input_video: Path, output_video: Path) -> None:
    """Stabilize shaky footage using VidStab."""
    stabilizer = VidStab()
    stabilizer.stabilize(str(input_video), str(output_video))


def _frames_from_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def _write_frames(frames, original_path: Path, output_path: Path, fps: float) -> None:
    h, w = frames[0].shape[:2]
    tmp = Path(output_path).with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    # copy audio from original file
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(tmp),
        "-i",
        str(original_path),
        "-map",
        "0:v",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        str(output_path),
    ])
    tmp.unlink(missing_ok=True)


def focus_speaker(input_video: Path, output_video: Path) -> None:
    """Crop to the primary face and keep it centered."""
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for frame in _frames_from_video(input_video):
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            margin = 20
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            w = min(w + 2 * margin, width - x)
            h = min(h + 2 * margin, height - y)
            cropped = frame[y:y + h, x:x + w]
            resized = cv2.resize(cropped, (width, height))
            frames.append(resized)
        else:
            frames.append(frame)
    cap.release()
    if frames:
        _write_frames(frames, input_video, output_video, fps)


def blur_bg(input_video: Path, output_video: Path) -> None:
    """Blur the background while keeping the subject sharp using Selfie Segmentation."""
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for frame in _frames_from_video(input_video):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_selfie.process(rgb)
        mask = results.segmentation_mask
        mask = cv2.resize(mask, (width, height))
        mask_3c = cv2.merge([mask, mask, mask])
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        composite = frame * mask_3c + blurred * (1 - mask_3c)
        composite = composite.astype("uint8")
        frames.append(composite)
    cap.release()
    if frames:
        _write_frames(frames, input_video, output_video, fps)


def adjust_levels(input_video: Path, output_video: Path) -> None:
    """Apply automatic contrast and colour adjustment."""
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for frame in _frames_from_video(input_video):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        frames.append(final)
    cap.release()
    if frames:
        _write_frames(frames, input_video, output_video, fps)
