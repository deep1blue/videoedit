# AI Video Fixer

AI-powered quick-fix video editor – turn a raw clip into a polished asset with one click. Upload a video, press the buttons you need (stabilise, denoise, blur background, etc.) and grab the processed file, all in a single Gradio interface.

---

## ✨ Features

| Button              | What it does                           | Under the hood                                   |
|---------------------|----------------------------------------|--------------------------------------------------|
| Stabilise           | Removes camera shake                   | OpenCV + VidStab                                  |
| Focus on Speaker    | Crops/zooms to face & smooth tracks    | MediaPipe Face Detection → smart crop            |
| Blur Background     | Keeps subject sharp, blurs rest        | MediaPipe Selfie-Segmentation + Gaussian blur    |
| Auto Contrast / Colour | Punchier look, fixes dull footage  | OpenCV CLAHE + ffmpeg `eq=`                      |
| Denoise Audio       | Cuts hiss, A/C hum, reverb             | RNNoise (default) or Demucs-light                |
| Normalise Volume    | Broadcast-safe loudness                | ffmpeg `loudnorm` (-14 LUFS)                     |
| Remove Filler Words | Cuts “um, uh, like” segments           | faster-whisper tiny-int8 + timestamp cuts        |
| Ask Video           | Answers questions about the uploaded video | SmolVLM2 video QA via Hugging Face |

All operations are functional – each button writes a fresh temp file so users can chain effects in any order.

---

## 🏗️ Architecture

```
video-fixer/
├── app.py                # Gradio routes & state
├── processing/
│   ├── video_utils.py    # stabilize(), blur_bg(), focus_speaker(), adjust_levels()
│   ├── audio_utils.py    # denoise(), normalize_gain(), cut_fillers()
│   ├── llm.py           # ask_video_question()
│   └── transcript.py     # get_transcript(), find_filler_words()
└── tmp/                  # ephemeral workspace
```

* ffmpeg handles all (de)muxing and re-encoding.
* Models are light (< 60 MB each) and downloaded once on first run.

---

## 📦 Requirements

| Runtime | Version | Notes |
|---------|---------|-------|
| Python  | 3.10 +  | Tested on 3.10 & 3.11 |
| PyTorch | ≥ 2.3   | CUDA optional |
| ffmpeg  | ≥ 6.0   | Install via Homebrew/apt |

Python libs (see `requirements.txt`): Gradio 4.*, opencv-python, mediapipe, vidstab, ffmpeg-python, pydub, faster-whisper, demucs, rnnoise-wrapper, nltk.
FastAPI is installed by Gradio and still expects Pydantic v1, so the
requirements file pins `pydantic<2` to avoid schema errors when the server
starts.

---

## 🚀 Quick Start (macOS/Linux)

```bash
# 1. System deps
brew install ffmpeg   # or: sudo apt-get install ffmpeg

# 2. Clone & install
git clone https://github.com/your-org/video-fixer.git
cd video-fixer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. One-off model downloads
python - <<'PY'
from demucs.pretrained import get_model; get_model('htdemucs_light')
from faster_whisper import WhisperModel; WhisperModel('tiny-int8', download_root='./models')
PY

# 4. Launch
python app.py  # => http://127.0.0.1:7860
```

Docker users:

```bash
docker build -t video-fixer .
docker run -p 7860:7860 video-fixer
```

---

## 🖥️ Using the App
1. Upload an MP4/MOV (≤ 300 MB).
2. Click any fix buttons – each re-renders and previews the clip.
3. Download the final MP4 via the built-in Gradio link.
4. Chain buttons in any order; last action wins.

Tip: double-click the video to view full-screen.

---

## 🧩 How It Works
1. `app.py` saves upload → `tmp/input.mp4` and spawns a new temp path per action.
2. Video actions (stabilise, blur, etc.) call `video_utils.py`, which leverages OpenCV frames in memory, then re-encodes with ffmpeg.
3. Audio actions (denoise, normalize) extract the audio track, process via Torch or RNNoise C library, then merge back.
4. Filler-word removal:
   * Transcribe with Whisper-tiny-int8 (≈ 30 MB, real-time CPU).
   * Parse transcript for filler-word tokens (`nltk.corpus.stopwords` + `CUSTOM_LIST`).
   * Use ffmpeg concat demuxer to cut those time ranges.
5. Output is served back to Gradio and streamed in the browser.

Pipeline is modular – add new buttons by dropping a function into `video_utils.py`/`audio_utils.py` and one line in `app.py`.

---

## 🗺️ Two-Week MVP Roadmap

| Day | Deliverable |
|----|------------|
| 1-2 | Repo scaffold, ffmpeg wrappers, Gradio upload/preview |
| 3-4 | Video stabiliser + auto contrast |
| 5-6 | Background blur & face-crop zoom |
| 7-8 | RNNoise / Demucs denoise + loudness |
| 9-10 | Whisper transcript + filler-cut |
| 11-12 | Button chaining, temp cleanup, Dockerfile |
| 13-14 | UI polish, README, demo screencast |

---

## 📚 References / Credits
* [VidStab](https://github.com/georgegach/vidstab)
* RNNoise – Xiph.Org Foundation
* Demucs v4 – Facebook AI Research
* faster-whisper – A. Méry
* MediaPipe Face Detection & Selfie Segmentation – Google Research
* SmolVLM2 – Apple ML Research

---

## 📜 License

MIT – see LICENSE file.

© 2025 AI Video Fixer Contributors. Happy editing! 🎬
