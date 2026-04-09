# Mic
#  ↓  (RNNOIse)
# WebRTC VAD
#  ↓
# Speech FSM
#  ↓
# Speech Segment Buffer
#  ↓
# Speaker Verifier  ← (resemblyzer)
#  ↓
# Whisper
#  ↓
# Keyword Matcher
#  ↓
# BERT Text Classifier
#  ↓
# Final Risk Score  (speaker-aware)
#  ↓
# Alert

import os
import time
import queue
import threading
import signal
import sys
import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad

from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline

try:
    from speaker_register import SpeakerVerifier
    SPEAKER_AVAILABLE = True
except ImportError:
    SPEAKER_AVAILABLE = False
    print("[WARN] speaker_register.py not found — speaker ID disabled")

# ============================================================
# CONFIG
# ============================================================

SR          = 16000
CHUNK_MS    = 30
CHUNK       = int(SR * CHUNK_MS / 1000)  
CHANNELS    = 1

# VAD
VAD_MODE    = 1   

# FSM
START_SPEECH_FRAMES = 3     
END_SILENCE_FRAMES  = 15    
MIN_SEGMENT_SEC     = 0.3
MAX_SEGMENT_SEC     = 6.0

# Whisper
WHISPER_MODEL   = "base.en"
WHISPER_DEVICE  = "cpu"
WHISPER_COMPUTE = "int8"

# BERT
BERT_MIN_TEXT_LEN = 3

# ============================================================
# ENUMS & DATACLASSES
# ============================================================

class AlertLevel(enum.Enum):
    NONE     = "NONE"
    CAUTION  = "CAUTION"
    WARNING  = "WARNING"
    ALERT    = "ALERT"
    CRITICAL = "CRITICAL"


@dataclass
class PipelineResult:
    alert_level:   AlertLevel    = AlertLevel.NONE
    fusion_score:  float         = 0.0
    yamnet_label:  str           = "—"
    yamnet_score:  float         = 0.0
    keyword_hit:   Optional[str] = None
    text_risk:     float         = 0.0
    transcript:    Optional[str] = None
    speaker_id:    Optional[str] = None   
    speaker_score: float         = 0.0   
    latency_ms:    float         = 0.0
    bert_label:    str           = "NEUTRAL"
    bert_score:    float         = 0.0
    keyword_score: float         = 0.0

    def to_dict(self):
        return {
            "alert_level":   self.alert_level.value,
            "fusion_score":  round(self.fusion_score, 3),
            "yamnet_label":  self.yamnet_label,
            "yamnet_score":  round(self.yamnet_score, 3),
            "keyword_hit":   self.keyword_hit,
            "keyword_score": round(self.keyword_score, 3),
            "text_risk":     round(self.text_risk, 3),
            "transcript":    self.transcript,
            "speaker_id":    self.speaker_id,
            "speaker_score": round(self.speaker_score, 3),
            "latency_ms":    round(self.latency_ms, 1),
            "bert_label":    self.bert_label,
            "bert_score":    round(self.bert_score, 3),
        }


@dataclass
class VehicleContext:
    speed_kmh:       float = 0.0
    eye_openness:    float = 1.0
    gaze_deviation:  float = 0.0


# ============================================================
# GLOBALS (standalone CLI mode only)
# ============================================================

audio_q       = queue.Queue()
running       = True
chunk_counter = 0


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO STATUS] {status}")
    audio_q.put(indata.copy().reshape(-1))


# ============================================================
# HELPERS
# ============================================================

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)

def float_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16).tobytes()

def normalize_audio(x: np.ndarray, target_rms=0.05, max_gain=8.0):
    cur = rms(x)
    if cur < 1e-6:
        return x
    gain = min(target_rms / cur, max_gain)
    return np.clip(x * gain, -1.0, 1.0)


def _is_whisper_hallucination(text: str) -> bool:
    """Detect Whisper hallucinations — repetitive loops on noisy audio."""
    if not text or len(text) < 20:
        return False
    t = text.lower().strip()
    # 1. Check for any 4-8 word phrase repeated 3+ times
    words = t.split()
    for ngram_len in range(3, 9):
        if len(words) < ngram_len * 3:
            continue
        for start in range(len(words) - ngram_len * 3 + 1):
            phrase = " ".join(words[start:start + ngram_len])
            count = t.count(phrase)
            if count >= 3:
                return True
    # 2. Very long transcript from a short segment is suspicious
    if len(text) > 600:
        return True
    return False


# ============================================================
# KEYWORD SPOTTER
# ============================================================

class KeywordSpotter:
    def __init__(self):
        self.keywords = {
            "help":             1.00,
            "emergency":        1.00,
            "accident":         0.95,
            "crash":            0.95,
            "sleepy":           0.90,
            "drowsy":           0.90,
            "tired":            0.75,
            "can't drive":      0.95,
            "cannot drive":     0.95,
            "not feeling well": 0.85,
            "call ambulance":   1.00,
            "call police":      0.95,
            "save me":          1.00,
            "i need help":      1.00,
            "i am sleepy":      0.95,
            "i feel sleepy":    0.95,
        }

    def score_text(self, text: str):
        text = text.lower().strip()
        best_kw, best_score = None, 0.0
        for kw, score in self.keywords.items():
            if kw in text and score > best_score:
                best_kw, best_score = kw, score
        return best_kw, best_score


# ============================================================
# BERT TEXT CLASSIFIER
# ============================================================

class TextRiskClassifier:
    def __init__(self):
        print("[INFO] Loading BERT text classifier...")
        self.pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        print("[OK] BERT text classifier loaded")

    def classify(self, text: str):
        text = text.strip()
        if len(text) < BERT_MIN_TEXT_LEN:
            return {"label": "NEUTRAL", "score": 0.0, "risk": 0.0}
        try:
            out   = self.pipe(text, truncation=True)[0]
            label = out["label"]
            score = float(out["score"])
            risk  = score if label == "NEGATIVE" else 0.15 * (1.0 - score)
            return {"label": label, "score": score, "risk": risk}
        except Exception as e:
            print(f"[BERT ERROR] {e}")
            return {"label": "ERROR", "score": 0.0, "risk": 0.0}


# ============================================================
# SPEECH FSM
# ============================================================

class SpeechFSM:
    def __init__(self):
        self.in_speech  = False
        self.speech_run = 0
        self.silence_run = 0

    def update(self, is_speech: bool) -> str:
        if is_speech:
            self.speech_run  += 1
            self.silence_run  = 0
        else:
            self.silence_run += 1
            self.speech_run   = 0

        if not self.in_speech and self.speech_run >= START_SPEECH_FRAMES:
            self.in_speech = True
            return "start"
        if self.in_speech and is_speech:
            return "continue"
        if self.in_speech and self.silence_run >= END_SILENCE_FRAMES:
            self.in_speech = False
            return "end"
        return "idle"


# ============================================================
# ASR
# ============================================================

class ASR:
    def __init__(self):
        print("[INFO] Loading Whisper...")
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )
        print("[OK] faster-whisper loaded")

    def transcribe(self, audio: np.ndarray):
        t0 = time.time()
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            beam_size=1,
            vad_filter=False,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
        )
        text       = " ".join(seg.text.strip() for seg in segments).strip()
        latency_ms = (time.time() - t0) * 1000.0
        return text, latency_ms


# ============================================================
# FINAL RISK  (speaker-aware)
# ============================================================

def compute_final_risk(keyword_score: float,
                       bert_risk: float,
                       speaker_id: str = "UNKNOWN") -> tuple:

    base = max(keyword_score, bert_risk)

    # High-confidence keywords bypass speaker weighting entirely
    KEYWORD_OVERRIDE_THRESHOLD = 0.90
    if keyword_score >= KEYWORD_OVERRIDE_THRESHOLD:
        weighted = base   # full weight — "help", "emergency", "call ambulance"
    elif speaker_id == "DRIVER":
        weighted = base * 1.0
    elif speaker_id == "PASSENGER":
        weighted = base * 0.7
    else:  # UNKNOWN
        weighted = base * 0.6   # raised from 0.4 — don't suppress real alerts

    if weighted >= 0.85:
        alert = "CRITICAL"
    elif weighted >= 0.60:
        alert = "WARNING"
    elif weighted >= 0.35:
        alert = "CAUTION"
    else:
        alert = "NONE"

    return weighted, alert


# ============================================================
# ALERT OUTPUT
# ============================================================

class AlertOutput:
    """Dispatches PipelineResult to registered callbacks."""
    def dispatch(self, result: PipelineResult):
        level = result.alert_level.value
        if level != "NONE":
            print(
                f"[ALERT] {level} — speaker={result.speaker_id} "
                f"risk={result.fusion_score:.2f} "
                f"kw={result.keyword_hit} "
                f"transcript={result.transcript!r}"
            )


# ============================================================
# DMS PIPELINE
# ============================================================

class DMSPipeline:
    """
    Class-based API used by app.py.
    Mic → VAD → FSM → SpeakerVerifier → ASR → Keyword → BERT → Risk → Alert
    """

    def __init__(self, mic_device=None):
        self.mic_device     = mic_device
        self._running       = False
        self._thread        = None
        self._lock          = threading.Lock()
        self._latest_result = PipelineResult()
        self._vehicle_ctx   = VehicleContext()
        self._alert_output  = AlertOutput()

        # Sub-components (lazy-loaded in _run)
        self._vad      = None
        self._kws      = None
        self._bert     = None
        self._asr      = None
        self._fsm      = None
        self._speaker  = None   # SpeakerVerifier instance

    # ── Public API ────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def update_vehicle_context(self, ctx: VehicleContext):
        with self._lock:
            self._vehicle_ctx = ctx

    def enrol_driver(self, seconds: float = 5.0):
        """
        Record driver voice and save embedding.
        Called from app.py POST /audio_enrol endpoint.
        Delegates to SpeakerVerifier.enrol().
        """
        if self._speaker is None:
            print("[ENROL] Speaker module not loaded yet — retrying in 1s")
            time.sleep(1.0)
            if self._speaker is None:
                print("[ENROL] Speaker module unavailable")
                return

        print(f"[ENROL] Recording driver voice for {seconds}s — speak now!")
        frames = []

        def _cb(indata, f, t, s):
            frames.append(indata.copy().reshape(-1))

        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=CHUNK, callback=_cb):
            time.sleep(seconds)

        audio = np.concatenate(frames)
        self._speaker.enrol(audio)
        print("[ENROL] Driver voice registered ✓")

    def get_latest_result(self) -> PipelineResult:
        with self._lock:
            return self._latest_result

    def get_speaker_status(self) -> dict:
        """Return speaker registration state for /status endpoint."""
        if self._speaker is None:
            return {"enrolled": False, "available": False}
        info = self._speaker.status()
        info["available"] = True
        return info

    # ── Internal loop ─────────────────────────────────────────

    def _run(self):
        print("[DMS] Loading sub-components…")
        self._vad  = webrtcvad.Vad(VAD_MODE)
        self._kws  = KeywordSpotter()
        self._bert = TextRiskClassifier()
        self._asr  = ASR()
        self._fsm  = SpeechFSM()

        # Load speaker verifier
        if SPEAKER_AVAILABLE:
            try:
                self._speaker = SpeakerVerifier()
                print("[DMS] Speaker verifier ready ✓")
            except Exception as e:
                print(f"[DMS] Speaker verifier failed to load: {e}")
                self._speaker = None
        else:
            self._speaker = None

        local_q         = queue.Queue()
        current_segment = []

        def _cb(indata, frames, time_info, status):
            if status:
                print(f"[AUDIO STATUS] {status}")
            local_q.put(indata.copy().reshape(-1))

        print("[DMS] Opening mic stream…")
        try:
            stream = sd.InputStream(
                samplerate=SR,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK,
                device=self.mic_device,
                callback=_cb,
            )
            stream.start()
        except Exception as e:
            print(f"[DMS] Mic open failed: {e}")
            self._running = False
            return

        print("[DMS] Audio pipeline running ✓")

        while self._running:
            try:
                chunk = local_q.get(timeout=0.2)
            except queue.Empty:
                continue

            raw   = chunk.astype(np.float32)
            clean = normalize_audio(raw, target_rms=0.05)

            pcm = float_to_pcm16(raw)
            try:
                is_speech = self._vad.is_speech(pcm, SR)
            except Exception:
                is_speech = False

            state = self._fsm.update(is_speech)

            if state == "start":
                current_segment = [clean.copy()]
            elif state == "continue":
                current_segment.append(clean.copy())
                if len(np.concatenate(current_segment)) / SR >= MAX_SEGMENT_SEC:
                    state = "end"

            if state == "end" and len(current_segment) > 0:
                segment         = np.concatenate(current_segment)
                current_segment = []

                if len(segment) / SR < MIN_SEGMENT_SEC:
                    continue

                # ── Speaker ID ────────────────────────────────
                if self._speaker is not None:
                    speaker_id, spk_score = self._speaker.identify(segment)
                else:
                    speaker_id, spk_score = "UNKNOWN", 0.0

                # ── ASR ───────────────────────────────────
                text, asr_latency = self._asr.transcribe(segment)

                # ── Hallucination filter ──────────────────
                if text and _is_whisper_hallucination(text):
                    print(f"[DMS] Hallucination filtered: {text[:80]!r}…")
                    text = ""

                # ── Keyword ───────────────────────────────────
                kw, kw_score = self._kws.score_text(text)

                # ── BERT ──────────────────────────────────────
                bert_out = self._bert.classify(text)

                # ── Risk (speaker-aware) ──────────────────────
                final_risk, alert_str = compute_final_risk(
                    kw_score, bert_out["risk"], speaker_id
                )

                alert_level = AlertLevel[alert_str]

                result = PipelineResult(
                    alert_level   = alert_level,
                    fusion_score  = final_risk,
                    yamnet_label  = "speech",
                    yamnet_score  = 0.0,
                    keyword_hit   = kw,
                    keyword_score = kw_score,
                    text_risk     = bert_out["risk"],
                    transcript    = text if text else None,
                    speaker_id    = speaker_id,
                    speaker_score = spk_score,
                    latency_ms    = asr_latency,
                    bert_label    = bert_out["label"],
                    bert_score    = bert_out["score"],
                )

                with self._lock:
                    self._latest_result = result

                self._alert_output.dispatch(result)

                print(
                    f"[DMS] speaker={speaker_id}({spk_score:.2f}) "
                    f"transcript={text!r} kw={kw} "
                    f"risk={final_risk:.2f} alert={alert_str}"
                )

        stream.stop()
        stream.close()
        print("[DMS] Audio pipeline stopped")


# ============================================================
# SHUTDOWN
# ============================================================

def stop_handler(sig, frame):
    global running
    print("\n[INFO] Stopping...")
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT,  stop_handler)
signal.signal(signal.SIGTERM, stop_handler)


# ============================================================
# STANDALONE CLI MAIN
# ============================================================

def main():
    global running, chunk_counter

    vad = webrtcvad.Vad(VAD_MODE)
    kws = KeywordSpotter()
    bert = TextRiskClassifier()
    asr  = ASR()
    fsm  = SpeechFSM()

    speaker = None
    if SPEAKER_AVAILABLE:
        try:
            speaker = SpeakerVerifier()
        except Exception as e:
            print(f"[WARN] SpeakerVerifier failed: {e}")

    current_segment = []

    print("\n🎤 Speak near the mic. Try:")
    print("   'I am feeling sleepy'  /  'Help me'  /  'I cannot drive'")
    print("\n[OK] Mic stream started. Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SR, channels=CHANNELS, dtype="float32",
        blocksize=CHUNK, callback=audio_callback
    ):
        while running:
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            chunk_counter += 1
            raw   = chunk.astype(np.float32)
            clean = normalize_audio(raw, target_rms=0.05)

            pcm = float_to_pcm16(raw)
            is_speech = vad.is_speech(pcm, SR)
            state     = fsm.update(is_speech)

            if state == "start":
                current_segment = [clean.copy()]
            elif state == "continue":
                current_segment.append(clean.copy())
                if len(np.concatenate(current_segment)) / SR >= MAX_SEGMENT_SEC:
                    state = "end"

            if chunk_counter % 10 == 0:
                print(
                    f"[CHUNK {chunk_counter:04d}] "
                    f"rms={rms(raw):.5f} speech={is_speech} fsm={state}"
                )

            if state == "end" and len(current_segment) > 0:
                segment         = np.concatenate(current_segment)
                current_segment = []

                if len(segment) / SR < MIN_SEGMENT_SEC:
                    print("\n[SKIP] segment too short\n")
                    continue

                print("\n================ DMS AUDIO DEBUG ================")

                # Speaker
                if speaker is not None:
                    spk_id, spk_score = speaker.identify(segment)
                    print(f"[SPEAKER]    id={spk_id}  score={spk_score:.3f}")
                else:
                    spk_id, spk_score = "UNKNOWN", 0.0

                # ASR
                text, latency = asr.transcribe(segment)
                print(f"[ASR]        transcript={repr(text)}  latency={latency:.1f}ms")

                # Keyword
                kw, kw_score = kws.score_text(text)
                print(f"[KEYWORD]    kw={kw}  score={kw_score:.2f}")

                # BERT
                bout = bert.classify(text)
                print(f"[BERT]       label={bout['label']}  score={bout['score']:.2f}  risk={bout['risk']:.2f}")

                # Risk
                risk, alert = compute_final_risk(kw_score, bout["risk"], spk_id)
                print(f"[RISK]       score={risk:.2f}  alert={alert}")
                print("=================================================\n")


if __name__ == "__main__":
    main()
