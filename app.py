import cv2
import json
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, Response, render_template, jsonify, request

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARN] onnxruntime not installed. Vision in DEMO mode.")

try:
    from dms_pipeline import (
        DMSPipeline, VehicleContext, AlertLevel, PipelineResult,
    )
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"[WARN] dms_pipeline not found ({e}). Audio disabled.")

try:
    from face_verification import (
        load_recognition_model as load_face_model,
        get_largest_face, passive_liveness_check,
        save_embedding, load_embedding, compare_embeddings,
        save_preview, DB_DIR, PREVIEW_DIR,
    )
    FACE_AVAILABLE = True
except ImportError as e:
    FACE_AVAILABLE = False
    print(f"[WARN] face_verification not found ({e}). Face ID disabled.")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "edge_driver_model.onnx"
IMG_SIZE     = 224
CAMERA_INDEX = 0
STREAM_FPS   = 25
JPEG_QUALITY = 80

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────────────────────────────────────
TASK_META = {
    "drowsiness": {
        "label": "Drowsiness", "icon": "😴",
        "classes": {0: "Alert", 1: "Drowsy"},
        "alert_class": 1, "alert_threshold": 0.65,
        "color": {"Alert": "#00e5a0", "Drowsy": "#ff4444"},
    },
    "gaze": {
        "label": "Gaze Direction", "icon": "👁",
        "classes": {
            0: "Bottom-Left",  1: "Middle-Left",  2: "Top-Left",
            3: "Bottom-Right", 4: "Middle-Right", 5: "Top-Right",
            6: "Top-Center",   7: "Bottom-Center",
        },
        "alert_class": None, "color": {},
    },
    "yawn": {
        "label": "Yawn Detection", "icon": "🥱",
        "classes": {0: "No Yawn", 1: "Yawning"},
        "alert_class": 1, "alert_threshold": 0.70,
        "color": {"No Yawn": "#00e5a0", "Yawning": "#ffaa00"},
    },
    "emotion": {
        "label": "Emotion", "icon": "🎭",
        "classes": {
            0: "Angry", 1: "Disgust", 2: "Fear",
            3: "Happy", 4: "Sad",    5: "Surprise", 6: "Neutral",
        },
        "alert_class": None,
        "color": {
            "Angry": "#ff4444", "Disgust": "#cc44ff",
            "Fear":  "#ff8800", "Happy":   "#00e5a0",
            "Sad":   "#4488ff", "Surprise":"#ffdd00",
            "Neutral":"#aaaaaa",
        },
    },
    "eye_state": {
        "label": "Eye State", "icon": "👀",
        "classes": {0: "Open", 1: "Closed"},
        "alert_class": 1, "alert_threshold": 0.80,
        "color": {"Open": "#00e5a0", "Closed": "#ff4444"},
    },
    "activity": {
        "label": "Activity", "icon": "🚗",
        "classes": {
            0: "Safe Driving", 1: "Distracted",
            2: "Phone Use",    3: "Drinking",
        },
        "alert_class": None, "alert_threshold": 0.60,
        "color": {
            "Safe Driving": "#00e5a0", "Distracted": "#ffaa00",
            "Phone Use":    "#ff4444", "Drinking":   "#ff4444",
        },
    },
}

TASK_ORDER = ["drowsiness", "eye_state", "yawn", "gaze", "emotion", "activity"]

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
state_lock         = threading.Lock()
latest_frame_jpg   = None
latest_predictions = {}
latest_fps         = 0.0
camera_ok          = False

# Audio state — updated by audio pipeline callback
audio_lock          = threading.Lock()
latest_audio_result = {
    "active":         False,
    "level":          "NONE",
    "fusion_score":   0.0,
    "yamnet_label":   "—",
    "yamnet_score":   0.0,
    "keyword":        None,
    "keyword_score":  0.0,
    "text_risk":      0.0,
    "transcript":     None,
    "speaker":        None,          # "DRIVER" | "PASSENGER" | "UNKNOWN"
    "speaker_score":  0.0,           # cosine similarity
    "latency_ms":     0.0,
    "bert_label":     "NEUTRAL",
    "bert_score":     0.0,
}
audio_event_queue = deque(maxlen=50)
audio_pipeline    = None
audio_ok          = False

# Enrollment progress state
enrol_state = {
    "phase":    "idle",    # idle | recording | processing | done | error
    "progress": 0,         # 0-100
    "message":  "",
    "driver_name": None,
}

# ── Face verification state ──────────────────────────────────────────────────
face_lock       = threading.Lock()
face_app_model  = None          # InsightFace model (lazy loaded)
latest_raw_frame = None         # Raw BGR frame from camera thread

face_enrol_state = {
    "phase":       "idle",     # idle | capturing | processing | done | error
    "progress":    0,
    "message":     "",
    "driver_name": None,
}

face_verify_state = {
    "active":          False,
    "match":           False,
    "similarity":      0.0,
    "liveness_label":  "—",
    "liveness_score":  0.0,
    "driver_name":     None,
}
face_verify_running = False     # Controls verify background loop

# ─────────────────────────────────────────────────────────────────────────────
# ONNX SESSION
# ─────────────────────────────────────────────────────────────────────────────
session = None

def load_model():
    global session
    if not ONNX_AVAILABLE:
        return
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"[OK] Vision model loaded: {MODEL_PATH}")
        print(f"     Provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"[WARN] Vision model not loaded ({e}). Demo mode.")
        session = None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING + SOFTMAX
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb  = cv2.cvtColor(cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return norm.transpose(2, 0, 1)[np.newaxis]

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

# ─────────────────────────────────────────────────────────────────────────────
# DEMO PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
_demo_t = 0.0

def demo_predictions() -> dict:
    global _demo_t
    _demo_t += 0.04
    t = _demo_t
    results = {}
    for task, meta in TASK_META.items():
        n     = len(meta["classes"])
        base  = np.array([0.5 + 0.4 * np.sin(t + i * 1.3) for i in range(n)], dtype=np.float32)
        probs = softmax(base)
        pred  = int(np.argmax(probs))
        results[task] = {
            "pred":       pred,
            "label":      meta["classes"][pred],
            "confidence": float(probs[pred]),
            "probs":      {meta["classes"][i]: float(p) for i, p in enumerate(probs)},
        }
    return results

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────
_buffers: dict = {task: deque(maxlen=10) for task in TASK_META}

def smooth_predictions(raw: dict) -> dict:
    smoothed = {}
    for task, result in raw.items():
        probs_arr = np.array(list(result["probs"].values()), dtype=np.float32)
        _buffers[task].append(probs_arr)
        avg  = np.mean(_buffers[task], axis=0)
        pred = int(np.argmax(avg))
        meta = TASK_META[task]
        smoothed[task] = {
            "pred":       pred,
            "label":      meta["classes"][pred],
            "confidence": float(avg[pred]),
            "probs":      {meta["classes"][i]: float(p) for i, p in enumerate(avg)},
        }
    return smoothed

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PIPELINE — init & callback
# ─────────────────────────────────────────────────────────────────────────────
def _on_audio_result(result: "PipelineResult"):
    """Called by DMSPipeline on every pipeline result."""
    with audio_lock:
        latest_audio_result.update({
            "active":        result.alert_level != AlertLevel.NONE,
            "level":         result.alert_level.name,
            "fusion_score":  round(result.fusion_score, 3),
            "yamnet_label":  result.yamnet_label,
            "yamnet_score":  round(result.yamnet_score, 3),
            "keyword":       result.keyword_hit,
            "keyword_score": round(getattr(result, "keyword_score", 0.0), 3),
            "text_risk":     round(result.text_risk, 3),
            "transcript":    result.transcript,
            "speaker":       result.speaker_id,          # DRIVER/PASSENGER/UNKNOWN
            "speaker_score": round(getattr(result, "speaker_score", 0.0), 3),
            "latency_ms":    round(result.latency_ms, 1),
            "bert_label":    getattr(result, "bert_label", "NEUTRAL"),
            "bert_score":    round(getattr(result, "bert_score", 0.0), 3),
        })

        # Push non-NONE events to SSE alert log
        if result.alert_level != AlertLevel.NONE:
            audio_event_queue.append({
                "ts":      time.strftime("%H:%M:%S"),
                "level":   result.alert_level.name,
                "score":   round(result.fusion_score, 3),
                "label":   result.yamnet_label,
                "kw":      result.keyword_hit,
                "text":    result.transcript,
                "speaker": result.speaker_id,   # ← now included in event log
            })


def start_audio_pipeline():
    """Launch DMS audio pipeline in background."""
    global audio_pipeline, audio_ok

    if not AUDIO_AVAILABLE:
        print("[WARN] Audio pipeline not available.")
        return

    try:
        from dms_pipeline import AlertOutput, AlertLevel as AL

        original_dispatch = AlertOutput.dispatch

        def patched_dispatch(self, result):
            original_dispatch(self, result)
            _on_audio_result(result)   # forward all results to Flask state

        AlertOutput.dispatch = patched_dispatch

        audio_pipeline = DMSPipeline(mic_device=None)
        audio_pipeline.start()
        audio_ok = True
        print("[OK] Audio pipeline started")

        # Log speaker registration state on startup
        spk = audio_pipeline.get_speaker_status()
        if spk.get("enrolled"):
            print(f"[OK] Driver voiceprint loaded from '{spk.get('embed_path')}'")
        else:
            print("[WARN] No driver voiceprint enrolled. Run: python speaker_register.py")

    except Exception as e:
        print(f"[WARN] Audio pipeline failed to start: {e}")
        audio_ok = False


# ─────────────────────────────────────────────────────────────────────────────
# FRAME OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def _draw_overlay(frame: np.ndarray, preds: dict) -> np.ndarray:
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "DMS LIVE", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 160), 2)

    # Drowsiness
    drow = preds.get("drowsiness", {})
    if drow.get("label") == "Drowsy" and drow.get("confidence", 0) > 0.65:
        cv2.rectangle(frame, (0, h - 56), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, "DROWSINESS DETECTED",
                    (w // 2 - 140, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Eye state
    eye = preds.get("eye_state", {})
    if eye.get("label") == "Closed" and eye.get("confidence", 0) > 0.80:
        cv2.putText(frame, "EYES CLOSED", (w - 200, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    # Audio alert strip
    with audio_lock:
        audio = dict(latest_audio_result)

    if audio.get("active") and audio.get("level") in ("ALERT", "CRITICAL"):
        cv2.rectangle(frame, (0, 52), (w, 86), (0, 80, 220), -1)
        spk = audio.get("speaker", "")
        kw  = audio.get("keyword", "")
        txt = f"AUDIO [{spk}]: {audio.get('yamnet_label', '')}"
        if kw:
            txt += f" | KW: {kw}"
        cv2.putText(frame, txt, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 80), 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA + VISION THREAD
# ─────────────────────────────────────────────────────────────────────────────
def _blank_frame() -> np.ndarray:
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(f, "NO CAMERA", (220, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
    return f


def camera_thread():
    global latest_frame_jpg, latest_predictions, latest_fps, camera_ok

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    camera_ok = cap.isOpened()
    if not camera_ok:
        print("[WARN] Camera not found. Demo mode.")

    frame_times = deque(maxlen=30)

    while True:
        t0 = time.time()

        if camera_ok:
            ret, frame = cap.read()
            if not ret:
                camera_ok = False
                frame = _blank_frame()
        else:
            frame = _blank_frame()
            time.sleep(1 / STREAM_FPS)

        if session is not None and camera_ok:
            try:
                inp  = preprocess(frame)
                outs = session.run(None, {"image": inp})
                raw  = {}
                for i, task in enumerate(list(TASK_META.keys())):
                    probs = softmax(outs[i][0])
                    pred  = int(np.argmax(probs))
                    meta  = TASK_META[task]
                    raw[task] = {
                        "pred":       pred,
                        "label":      meta["classes"][pred],
                        "confidence": float(probs[pred]),
                        "probs":      {meta["classes"][j]: float(p)
                                       for j, p in enumerate(probs)},
                    }
                preds = smooth_predictions(raw)
            except Exception as e:
                print(f"[ERR] Vision inference: {e}")
                preds = demo_predictions()
        else:
            preds = demo_predictions()

        # Feed vision context into audio pipeline
        if audio_pipeline is not None and audio_ok:
            try:
                eye_conf = preds.get("eye_state", {}).get("confidence", 1.0)
                eye_open = 1.0 - eye_conf if preds.get("eye_state", {}).get("label") == "Closed" else eye_conf
                ctx = VehicleContext(
                    speed_kmh      = 60.0,
                    eye_openness   = float(eye_open),
                    gaze_deviation = 0.0,
                )
                audio_pipeline.update_vehicle_context(ctx)
            except Exception:
                pass

        annotated = _draw_overlay(frame.copy(), preds)
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) \
              if len(frame_times) > 1 else 0.0

        with state_lock:
            latest_frame_jpg   = jpg.tobytes()
            latest_predictions = preds
            latest_fps         = round(fps, 1)

        # Share raw frame for face verification
        with face_lock:
            global latest_raw_frame
            latest_raw_frame = frame.copy() if camera_ok else None

        elapsed = time.time() - t0
        time.sleep(max(0, (1 / STREAM_FPS) - elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           task_meta=json.dumps(TASK_META),
                           task_order=json.dumps(TASK_ORDER))


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                jpg = latest_frame_jpg
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(1 / STREAM_FPS)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predictions")
def predictions():
    """SSE: vision predictions + latest audio result fused."""
    def event_stream():
        last_sent = None
        while True:
            with state_lock:
                preds = dict(latest_predictions)
                fps   = latest_fps
            with audio_lock:
                audio = dict(latest_audio_result)

            payload = json.dumps({"predictions": preds, "fps": fps, "audio": audio})
            if payload != last_sent:
                last_sent = payload
                yield f"data: {payload}\n\n"
            time.sleep(1 / 15)

    return Response(event_stream(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.route("/audio_events")
def audio_events():
    """SSE: audio alert log (includes speaker field)."""
    def event_stream():
        last_idx = 0
        while True:
            with audio_lock:
                events = list(audio_event_queue)
            for ev in events[last_idx:]:
                yield f"data: {json.dumps(ev)}\n\n"
            last_idx = len(events)
            time.sleep(0.25)

    return Response(event_stream(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.route("/status")
def status():
    """System health — includes speaker registration state."""
    with audio_lock:
        audio = dict(latest_audio_result)

    speaker_status = {}
    if audio_pipeline is not None:
        speaker_status = audio_pipeline.get_speaker_status()

    return jsonify({
        "model_loaded":     session is not None,
        "camera_ok":        camera_ok,
        "onnx_available":   ONNX_AVAILABLE,
        "model_path":       MODEL_PATH,
        "audio_ok":         audio_ok,
        "audio_available":  AUDIO_AVAILABLE,
        "audio_level":      audio.get("level", "NONE"),
        "speaker":          speaker_status,   # {enrolled, embed_path, threshold}
    })


@app.route("/audio_enrol", methods=["POST"])
def audio_enrol():
    """
    POST /audio_enrol
    Triggers 5-second driver voice re-enrolment via the live mic.
    The driver should speak naturally into the mic after calling this.
    """
    if not audio_ok or audio_pipeline is None:
        return jsonify({"ok": False, "msg": "Audio pipeline not running"}), 503

    data = request.get_json(silent=True) or {}
    seconds = float(data.get("seconds", 5.0))
    driver_name = data.get("driver_name", "Driver")

    def _do_enrol():
        with audio_lock:
            enrol_state.update(phase="recording", progress=0,
                               message="Recording — speak clearly",
                               driver_name=driver_name)
        try:
            # Simulate progress during recording
            import time as _t
            total = seconds
            step  = 0.3
            elapsed = 0.0
            # Start actual enrollment in sub-thread
            enrol_done = threading.Event()
            enrol_err  = [None]

            def _actual_enrol():
                try:
                    audio_pipeline.enrol_driver(seconds=total)
                except Exception as exc:
                    enrol_err[0] = exc
                finally:
                    enrol_done.set()

            threading.Thread(target=_actual_enrol, daemon=True).start()

            # Update progress while recording
            while not enrol_done.is_set() and elapsed < total + 3:
                pct = min(int((elapsed / total) * 90), 90)
                with audio_lock:
                    enrol_state["progress"] = pct
                    if elapsed < total:
                        enrol_state["message"] = f"Recording — {max(0, int(total - elapsed))}s left"
                    else:
                        enrol_state["phase"] = "processing"
                        enrol_state["message"] = "Processing voiceprint…"
                _t.sleep(step)
                elapsed += step

            enrol_done.wait(timeout=5)

            if enrol_err[0]:
                raise enrol_err[0]

            with audio_lock:
                enrol_state.update(phase="done", progress=100,
                                   message="Enrollment complete ✓")
        except Exception as e:
            with audio_lock:
                enrol_state.update(phase="error", progress=0,
                                   message=f"Enrollment failed: {e}")

    threading.Thread(target=_do_enrol, daemon=True).start()
    return jsonify({
        "ok":  True,
        "msg": f"Enrolling driver voice — speak for {seconds:.0f}s",
    })


@app.route("/enrol_status")
def enrol_status():
    """Enrollment progress for frontend polling."""
    with audio_lock:
        return jsonify(dict(enrol_state))


@app.route("/audio_status")
def audio_status():
    with audio_lock:
        return jsonify(latest_audio_result)


@app.route("/voice_results")
def voice_results():
    """Full voice pipeline analysis including speaker ID."""
    with audio_lock:
        return jsonify(dict(latest_audio_result))


# ─────────────────────────────────────────────────────────────────────────────
# FACE ENROLLMENT & VERIFICATION ROUTES
# ─────────────────────────────────────────────────────────────────────────────

def _get_face_model():
    """Lazy-load InsightFace model (CPU)."""
    global face_app_model
    if face_app_model is None:
        if not FACE_AVAILABLE:
            raise RuntimeError("face_verification module not available")
        face_app_model = load_face_model(cpu=True)
    return face_app_model


@app.route("/face_enrol", methods=["POST"])
def face_enrol():
    """
    POST /face_enrol  {"driver_name": "shivam"}
    Captures the current camera frame, detects face, runs liveness,
    and saves the face embedding.
    """
    if not FACE_AVAILABLE:
        return jsonify({"ok": False, "msg": "Face verification module not available"}), 503

    data = request.get_json(silent=True) or {}
    driver_name = data.get("driver_name", "Driver").strip() or "Driver"

    def _do_face_enrol():
        with face_lock:
            face_enrol_state.update(
                phase="capturing", progress=10,
                message="Initializing face model…",
                driver_name=driver_name,
            )
        try:
            fmodel = _get_face_model()

            with face_lock:
                face_enrol_state.update(progress=30, message="Looking for face…")

            # Grab several frames and pick best face
            best_face = None
            best_frame = None
            best_score = 0.0
            for attempt in range(15):
                time.sleep(0.3)
                with face_lock:
                    raw = latest_raw_frame
                    face_enrol_state["progress"] = min(30 + attempt * 4, 85)
                if raw is None:
                    continue
                face = get_largest_face(fmodel, raw)
                if face is not None and face.det_score > best_score:
                    best_face = face
                    best_frame = raw.copy()
                    best_score = face.det_score
                    with face_lock:
                        face_enrol_state["message"] = f"Face detected (score={face.det_score:.2f})"

            if best_face is None:
                with face_lock:
                    face_enrol_state.update(
                        phase="error", progress=0,
                        message="No face detected in camera. Please face the camera.",
                    )
                return

            # Liveness check
            with face_lock:
                face_enrol_state.update(progress=88, message="Checking liveness…")
            liv_label, liv_score, is_real = passive_liveness_check(
                best_frame, best_face.bbox
            )

            if not is_real:
                with face_lock:
                    face_enrol_state.update(
                        phase="error", progress=0,
                        message=f"Liveness FAILED ({liv_label}). Use a real face.",
                    )
                return

            # Save embedding
            with face_lock:
                face_enrol_state.update(progress=95, message="Saving face embedding…")

            save_embedding(driver_name, best_face.embedding)

            import cv2 as _cv2
            from face_verification import draw_face_box
            preview = best_frame.copy()
            draw_face_box(preview, best_face, f"Enrolled: {driver_name}", (0, 255, 0))
            save_preview(preview, f"enrolled_{driver_name}.jpg")

            with face_lock:
                face_enrol_state.update(
                    phase="done", progress=100,
                    message="Face enrollment complete ✓",
                )

        except Exception as e:
            with face_lock:
                face_enrol_state.update(
                    phase="error", progress=0,
                    message=f"Enrollment failed: {e}",
                )

    threading.Thread(target=_do_face_enrol, daemon=True).start()
    return jsonify({"ok": True, "msg": f"Face enrollment started for '{driver_name}'"})


@app.route("/face_enrol_status")
def face_enrol_status():
    with face_lock:
        return jsonify(dict(face_enrol_state))


@app.route("/face_verify_start", methods=["POST"])
def face_verify_start():
    """
    POST /face_verify_start  {"driver_name": "shivam"}
    Starts continuous face verification in background.
    """
    global face_verify_running

    if not FACE_AVAILABLE:
        return jsonify({"ok": False, "msg": "Face verification module not available"}), 503

    data = request.get_json(silent=True) or {}
    driver_name = data.get("driver_name", "Driver").strip() or "Driver"

    # Check if face is enrolled
    import os
    emb_path = os.path.join(DB_DIR, f"{driver_name}.npy")
    if not os.path.exists(emb_path):
        return jsonify({"ok": False, "msg": f"No face enrolled for '{driver_name}'"}), 404

    if face_verify_running:
        return jsonify({"ok": True, "msg": "Verification already running"})

    face_verify_running = True

    def _verify_loop():
        global face_verify_running
        try:
            fmodel = _get_face_model()
            enrolled_emb = load_embedding(driver_name)

            while face_verify_running:
                with face_lock:
                    raw = latest_raw_frame
                if raw is None:
                    time.sleep(0.3)
                    continue

                face = get_largest_face(fmodel, raw)
                if face is None:
                    with face_lock:
                        face_verify_state.update(
                            active=True, match=False,
                            similarity=0.0,
                            liveness_label="No Face",
                            liveness_score=0.0,
                            driver_name=driver_name,
                        )
                    time.sleep(0.3)
                    continue

                liv_label, liv_score, is_real = passive_liveness_check(
                    raw, face.bbox
                )

                if not is_real:
                    with face_lock:
                        face_verify_state.update(
                            active=True, match=False,
                            similarity=0.0,
                            liveness_label=liv_label,
                            liveness_score=round(liv_score, 3),
                            driver_name=driver_name,
                        )
                else:
                    sim, is_match = compare_embeddings(
                        face.embedding, enrolled_emb, 0.45
                    )
                    with face_lock:
                        face_verify_state.update(
                            active=True, match=is_match,
                            similarity=round(sim, 3),
                            liveness_label=liv_label,
                            liveness_score=round(liv_score, 3),
                            driver_name=driver_name,
                        )

                time.sleep(0.5)  # ~2 FPS for face verify

        except Exception as e:
            print(f"[ERR] Face verify loop: {e}")
        finally:
            face_verify_running = False
            with face_lock:
                face_verify_state["active"] = False

    threading.Thread(target=_verify_loop, daemon=True).start()
    return jsonify({"ok": True, "msg": f"Face verification started for '{driver_name}'"})


@app.route("/face_verify_status")
def face_verify_status():
    with face_lock:
        state = dict(face_verify_state)
    # Convert numpy types to native Python (numpy bool_ is not JSON serializable)
    state["active"]    = bool(state.get("active", False))
    state["match"]     = bool(state.get("match", False))
    state["similarity"] = float(state.get("similarity", 0.0))
    state["liveness_score"] = float(state.get("liveness_score", 0.0))
    return jsonify(state)


@app.route("/face_verify_stop", methods=["POST"])
def face_verify_stop():
    global face_verify_running
    face_verify_running = False
    with face_lock:
        face_verify_state["active"] = False
    return jsonify({"ok": True, "msg": "Face verification stopped"})


@app.route("/face_status")
def face_status_route():
    """Check if a face is enrolled."""
    import os
    enrolled_faces = []
    if FACE_AVAILABLE and os.path.exists(DB_DIR):
        enrolled_faces = [
            f.replace(".npy", "")
            for f in os.listdir(DB_DIR)
            if f.endswith(".npy")
        ]
    return jsonify({
        "available":      FACE_AVAILABLE,
        "enrolled":       len(enrolled_faces) > 0,
        "enrolled_names": enrolled_faces,
        "verify_active":  face_verify_running,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    start_audio_pipeline()

    cam_t = threading.Thread(target=camera_thread, daemon=True)
    cam_t.start()
    print("[OK] Camera thread started")
    print("[OK] Open http://localhost:5000")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
