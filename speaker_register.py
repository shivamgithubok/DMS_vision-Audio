import os
import time
import numpy as np
import sounddevice as sd

from resemblyzer import VoiceEncoder, preprocess_wav

# ── Config ─────────────────────────────────────────────────────────
SR              = 16000
DEFAULT_SECONDS = 5.0
EMBED_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "driver_embedding.npy")
THRESHOLD       = 0.50   

class SpeakerVerifier:
    """Thin wrapper around resemblyzer for the DMS pipeline."""

    def __init__(self, embed_path: str = EMBED_PATH, threshold: float = THRESHOLD):
        self.embed_path = embed_path
        self.threshold  = threshold

        print("[SPEAKER] Loading resemblyzer encoder…")
        self.encoder = VoiceEncoder(device="cpu")
        print("[SPEAKER] Encoder ready ✓")

        # Load existing driver embedding if available
        self.driver_embed = None
        if os.path.isfile(self.embed_path):
            self.driver_embed = np.load(self.embed_path)
            print(f"[SPEAKER] Driver embedding loaded from {self.embed_path}")
        else:
            print("[SPEAKER] No driver embedding found — enrol first")


    def enrol(self, audio: np.ndarray):
        """
        Compute and save driver voiceprint from raw float32 audio (16 kHz).
        """
        wav = preprocess_wav(audio, source_sr=SR)
        embed = self.encoder.embed_utterance(wav)
        np.save(self.embed_path, embed)
        self.driver_embed = embed
        print(f"[SPEAKER] Driver embedding saved → {self.embed_path}")

    def identify(self, audio: np.ndarray) -> tuple:
        if self.driver_embed is None:
            return ("UNKNOWN", 0.0)

        wav   = preprocess_wav(audio, source_sr=SR)
        embed = self.encoder.embed_utterance(wav)

        score = float(np.dot(self.driver_embed, embed) /
                    (np.linalg.norm(self.driver_embed) *
                        np.linalg.norm(embed) + 1e-9))

        if score >= self.threshold:
            label = "DRIVER"
        elif score >= 0.35:
            label = "PASSENGER"
        else:
            label = "UNKNOWN"

        print(f"[SPEAKER DEBUG] raw_score={score:.4f}  threshold={self.threshold}  label={label}")
        return (label, round(score, 3))
        
    
    def status(self) -> dict:
        return {
            "enrolled":    self.driver_embed is not None,
            "embed_path":  self.embed_path,
            "threshold":   self.threshold,
        }