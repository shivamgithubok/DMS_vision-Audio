#    Camera Frame                            
#          |                                  
#          V                                  
#     Face Detection                          
#     (InsightFace det_10g.onnx)              
#          |                                  
#          V                                  
#     Landmark Alignment                      
#     (2d106det / 1k3d68.onnx)                
#          |                                  
#          V                                  
#     Passive Liveness  (texture-based)       
#         |
#     | Signal 1: Laplacian sharp  (45%) |    
#     | Signal 2: LBP micro-tex    (35%) |    
#     | Signal 3: HSV saturation   (20%) |    
#         |
#     REAL  --> continue                      
#     SPOOF --> REJECT                        
#          |                                  
#          V  (only if REAL)                  
#     Face Recognition                        
#     (w600k_r50.onnx, 512-d embedding)       
#          |                                  
#          V                                  
#     Cosine Similarity   threshold=0.45      
#          |                                  
#          V                                  
#       MATCH / NO MATCH                      
#   
import os
import cv2
import time
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


# ============================================================
# CONFIG
# ============================================================
DB_DIR      = "face_db"
PREVIEW_DIR = "face_preview"
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


# ============================================================
# LOAD RECOGNITION MODEL
# ============================================================
def load_recognition_model(cpu=False):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1 if cpu else 0, det_size=(640, 640))
    print("[INFO] InsightFace buffalo_l loaded.")
    return app



def compute_lbp(gray):
    """
    Compute Local Binary Pattern manually (no scikit-image needed).
    Compares each pixel to its 8 neighbours, builds binary code.
    """
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    # 8 neighbour offsets (clockwise from top-left)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    center = gray[1:-1, 1:-1].astype(np.int32)

    for i, (dy, dx) in enumerate(offsets):
        neighbour = gray[1+dy:h-1+dy, 1+dx:w-1+dx].astype(np.int32)
        lbp |= ((neighbour >= center).astype(np.uint8) << i)

    return lbp


def passive_liveness_check(frame, bbox, debug=False):
    """
    Texture-based passive liveness detection.

    Parameters:
        frame : full BGR frame
        bbox  : [x1, y1, x2, y2] face bounding box
        debug : if True, prints individual signal scores

    Returns:
        label   : str   e.g. "REAL (0.82)" or "SPOOF (0.31)"
        score   : float 0.0-1.0  (higher = more likely real)
        is_real : bool
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp to frame bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return "NO FACE", 0.0, False

    # Resize to fixed size for consistent scoring
    face_resized = cv2.resize(face_crop, (128, 128))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)

    # Signal 1: Laplacian sharpness
    # Real faces: high variance (lots of detail)
    # Photos/screens: low variance (smooth, flat)
    lap_var   = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_score = min(lap_var / 500.0, 1.0)   # real ~200-800, spoof ~20-150

    # Signal 2: LBP micro-texture variance
    # Real skin has rich micro-texture -> high LBP std
    # Printed/screen face is smooth    -> low LBP std
    lbp       = compute_lbp(gray)
    lbp_std   = float(np.std(lbp))
    lbp_score = min(lbp_std / 80.0, 1.0)    # real ~55-90, spoof ~20-45

    # Signal 3: HSV saturation naturalness
    # Screens are oversaturated; prints are flat
    # Real skin has moderate, varied saturation
    sat       = hsv[:, :, 1].astype(np.float32)
    sat_mean  = float(np.mean(sat))
    sat_score = 1.0 - abs(sat_mean - 80.0) / 120.0
    sat_score = float(np.clip(sat_score, 0.0, 1.0))

    # Combined score (weighted)
    # Laplacian is strongest signal for Jetson Nano camera
    score = (0.45 * lap_score) + (0.35 * lbp_score) + (0.20 * sat_score)
    score = float(np.clip(score, 0.0, 1.0))

    LIVENESS_THRESHOLD = 0.40
    is_real = score >= LIVENESS_THRESHOLD

    if debug:
        print(f"  [Liveness] lap={lap_score:.3f}  lbp={lbp_score:.3f}  "
              f"sat={sat_score:.3f}  final={score:.3f}  -> "
              f"{'REAL' if is_real else 'SPOOF'}")

    label = f"{'REAL' if is_real else 'SPOOF'} ({score:.2f})"
    return label, score, is_real


# ============================================================
# FACE EXTRACTION
# ============================================================
def get_largest_face(app, frame):
    faces = app.get(frame)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))


# ============================================================
# DRAW HELPERS
# ============================================================
def draw_face_box(frame, face, label="", color=(0, 255, 0)):
    if face is None:
        return frame
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, max(30, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def draw_status(frame, lines, start_y=40, color=(255, 255, 0)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, start_y + i*32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame


# ============================================================
# EMBEDDING HELPERS
# ============================================================
def save_embedding(name, embedding):
    path = os.path.join(DB_DIR, f"{name}.npy")
    np.save(path, embedding)
    print(f"[INFO] Embedding saved -> {path}")


def load_embedding(name):
    path = os.path.join(DB_DIR, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embedding for '{name}' at {path}")
    return np.load(path)


def compare_embeddings(emb1, emb2, threshold=0.45):
    sim = cosine_similarity(emb1.reshape(1,-1), emb2.reshape(1,-1))[0][0]
    return float(sim), sim >= threshold


# ============================================================
# CAMERA
# ============================================================
def open_camera(camera_id=0, width=1280, height=720):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def save_preview(frame, name="preview.jpg"):
    cv2.imwrite(os.path.join(PREVIEW_DIR, name), frame)


# ============================================================
# PIPELINE DIAGRAM
# ============================================================
def print_pipeline():
    print("")


# ============================================================
# ENROLL MODE
# ============================================================
def enroll_mode(app, name, camera_id=0, debug=False):
    print("=" * 60)
    print(f"[ENROLL MODE] User: {name}")
    print(" Press 's' to capture  |  'q' to quit")
    print("=" * 60)

    cap = open_camera(camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = get_largest_face(app, frame)
        frame_disp = frame.copy()

        if face is None:
            draw_status(frame_disp, ["No Face Detected"], color=(0,0,255))
        else:
            liv_label, liv_score, is_real = passive_liveness_check(
                frame, face.bbox, debug=debug)

            color = (0,255,0) if is_real else (0,0,255)
            draw_face_box(frame_disp, face,
                          f"det:{face.det_score:.2f} | {liv_label}", color)
            draw_status(frame_disp, [
                f"Liveness : {liv_label}",
                "Press 's' to enroll | 'q' to quit"
            ], color=color)

        cv2.imshow("Enroll", frame_disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if face is None:
                print("[WARN] No face in frame.")
            elif not is_real:
                print(f"[WARN] Liveness FAILED ({liv_label}). Use a real face.")
            else:
                save_preview(frame_disp, f"enrolled_{name}.jpg")
                save_embedding(name, face.embedding)
                print(f"[OK] Enrolled '{name}' successfully.")
                break

        elif key == ord('q'):
            print("[INFO] Enroll cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# VERIFY MODE
# ============================================================
def verify_mode(app, name, threshold=0.45, camera_id=0, debug=False):
    print("=" * 60)
    print(f"[VERIFY MODE] User: {name}  threshold={threshold}")
    print(" Press 'q' to quit")
    print("=" * 60)

    enrolled_emb = load_embedding(name)
    cap = open_camera(camera_id)
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = get_largest_face(app, frame)
        frame_disp = frame.copy()

        now = time.time()
        fps = 1.0 / max(now - fps_time, 1e-6)
        fps_time = now

        if face is None:
            draw_status(frame_disp,
                        [f"FPS:{fps:.1f}", "No Face"], color=(0,0,255))
        else:
            # Step 1: Liveness check
            liv_label, liv_score, is_real = passive_liveness_check(
                frame, face.bbox, debug=debug)

            if not is_real:
                # Spoof — reject immediately, skip recognition
                color = (0, 0, 255)
                draw_face_box(frame_disp, face,
                              f"SPOOF ({liv_score:.2f})", color)
                draw_status(frame_disp, [
                    f"FPS:{fps:.1f}",
                    f"Liveness : {liv_label}",
                    "STATUS   : REJECTED (spoof)"
                ], color=color)

            else:
                # Step 2: Recognition (only runs if liveness passed)
                sim, is_match = compare_embeddings(
                    face.embedding, enrolled_emb, threshold)

                color = (0,255,0) if is_match else (0,165,255)
                draw_face_box(frame_disp, face,
                    f"{liv_label} | {'MATCH' if is_match else 'NO MATCH'} {sim:.3f}",
                    color)
                draw_status(frame_disp, [
                    f"FPS:{fps:.1f}  thr:{threshold}",
                    f"Liveness : {liv_label}",
                    f"Identity : {'MATCH' if is_match else 'NO MATCH'} (sim={sim:.3f})",
                ], color=color)

        cv2.imshow("Verify", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Verify stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Face Pipeline: Texture Liveness + InsightFace Recognition")
    parser.add_argument("--mode",      required=True, choices=["enroll", "verify"])
    parser.add_argument("--name",      required=True, help="User name")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--camera_id", type=int,   default=0)
    parser.add_argument("--cpu",       action="store_true")
    parser.add_argument("--debug",     action="store_true",
                        help="Print per-signal liveness scores to terminal")
    args = parser.parse_args()

    print_pipeline()
    print("=" * 60)
    print(f"Mode      : {args.mode}")
    print(f"Name      : {args.name}")
    print(f"Threshold : {args.threshold}")
    print(f"Camera    : {args.camera_id}")
    print(f"Device    : {'CPU' if args.cpu else 'GPU'}")
    print("=" * 60)

    app = load_recognition_model(cpu=args.cpu)

    if args.mode == "enroll":
        enroll_mode(app, args.name,
                    camera_id=args.camera_id, debug=args.debug)
    elif args.mode == "verify":
        verify_mode(app, args.name,
                    threshold=args.threshold,
                    camera_id=args.camera_id,
                    debug=args.debug)


if __name__ == "__main__":
    main()