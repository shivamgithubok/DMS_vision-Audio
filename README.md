# DMS Light — Driver Monitoring System (Light Model)

Real-time driver monitoring dashboard combining a **6-task ONNX vision model** with a **voice-based audio safety pipeline** — speaker verification, keyword detection, sentiment analysis, and multi-level alerting.

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For **Jetson** (CUDA/TensorRT):
```bash
pip install onnxruntime-gpu
```

### 3. Place your model

```bash
# Copy the trained ONNX model into this directory
cp /path/to/edge_driver_model.onnx ./
```

### 4. (Optional) Enrol driver voice

```bash
python speaker_register.py
# Speak for 5 seconds — saves driver_embedding.npy
```

## Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Dashboard

| Panel | Description |
|-------|-------------|
| Left | Live MJPEG webcam stream with HUD overlay |
| Right sidebar | 6-task predictions with per-class probability bars |
| Gaze grid | 3×3 direction indicator — active cell highlights |
| Audio panel | Speaker ID, transcript, BERT sentiment, alert level |
| Bottom | Alert log, uptime, alert counter |

## Vision Tasks

| Task | Classes |
|------|---------|
| Drowsiness | Alert, Drowsy |
| Eye State | Open, Closed |
| Yawn | No Yawn, Yawning |
| Gaze | 8 directions (Top/Middle/Bottom × Left/Right + Top-Center/Bottom-Center) |
| Emotion | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| Activity | Safe Driving, Distracted, Phone Use, Drinking |

## Audio Pipeline

The audio pipeline runs in a background thread and fuses with vision results:

1. **Mic capture** — 16 kHz, 30 ms chunks via `sounddevice`
2. **WebRTC VAD** — voice activity detection
3. **Speech FSM** — segments speech start/end boundaries
4. **Speaker verification** — `resemblyzer` cosine similarity against enrolled driver voiceprint
5. **Whisper ASR** — `faster-whisper` (base.en, int8) transcription
6. **Keyword spotter** — matches distress phrases ("help", "sleepy", "can't drive", etc.)
7. **BERT classifier** — `distilbert-base-uncased-finetuned-sst-2-english` sentiment → risk score
8. **Risk fusion** — speaker-weighted keyword + BERT risk → alert level (NONE / CAUTION / WARNING / ALERT / CRITICAL)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard page |
| `/video_feed` | GET | MJPEG video stream |
| `/predictions` | GET (SSE) | Vision predictions + audio state |
| `/audio_events` | GET (SSE) | Audio alert event log |
| `/status` | GET | System health + speaker registration state |
| `/audio_status` | GET | Latest audio pipeline result |
| `/voice_results` | GET | Full voice analysis (includes speaker ID) |
| `/audio_enrol` | POST | Trigger driver voice enrollment |
| `/enrol_status` | GET | Enrollment progress (polling) |

## Demo Mode

If the ONNX model or camera is not available, the app runs in **demo mode** with animated synthetic predictions — useful for verifying the UI without hardware.

## Jetson Deployment

```bash
# GPU inference
pip install onnxruntime-gpu
```
# DMS_vision-Audio
