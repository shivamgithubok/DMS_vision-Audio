import os
import numpy as np
import tempfile
import soundfile as sf

import nemo.collections.asr as nemo_asr

SR = 16000
EMBED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "driver_embedding.npy")

HIGH_THRESHOLD = 0.45   
LOW_THRESHOLD  = 0.30   


class SpeakerVerifier:
    """TitaNet-based speaker verification"""

    def __init__(self, embed_path: str = EMBED_PATH):
        self.embed_path = embed_path

        print("[SPEAKER] Loading TitaNet model...")
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
        print("[SPEAKER] TitaNet ready ✓")

        self.driver_embed = None

        if os.path.isfile(self.embed_path):
            self.driver_embed = np.load(self.embed_path)
            print(f"[SPEAKER] Driver embedding loaded from {self.embed_path}")
        else:
            print("[SPEAKER] No driver embedding found — enroll first")

    def _audio_to_embedding(self, audio: np.ndarray):
        """
        Convert numpy audio → temp wav → TitaNet embedding
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()

        
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / (np.max(np.abs(audio)) + 1e-9)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, SR)

        emb = self.model.get_embedding(temp_path)

        os.remove(temp_path)

        
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()

        return emb.flatten()

    def enrol(self, audio: np.ndarray):
        """
        Enroll driver voice
        """
        embed = self._audio_to_embedding(audio)

        np.save(self.embed_path, embed)
        self.driver_embed = embed

        print(f"[SPEAKER] Driver enrolled → {self.embed_path}")

    def identify(self, audio: np.ndarray):
        """
        Identify speaker
        """
        if self.driver_embed is None:
            return ("UNKNOWN", 0.0)

        embed = self._audio_to_embedding(audio)

        
        score = float(
            np.dot(self.driver_embed, embed) /
            (np.linalg.norm(self.driver_embed) *
             np.linalg.norm(embed) + 1e-9)
        )

        
        if score >= HIGH_THRESHOLD:
            label = "DRIVER"
        elif score >= LOW_THRESHOLD:
            label = "UNCERTAIN"
        else:
            label = "NOT_DRIVER"

        print(f"[SPEAKER DEBUG] score={score:.4f} → {label}")

        return (label, round(score, 3))

    def status(self):
        return {
            "enrolled": self.driver_embed is not None,
            "embed_path": self.embed_path,
            "high_threshold": HIGH_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
            "driver_name": os.path.splitext(os.path.basename(self.embed_path))[0]
                        if self.driver_embed is not None else None,
        }
