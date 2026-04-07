"""
detect.py – Real-Time Multi-Object Detection
HOG+SVM (MS-COCO trained) + MobileNet SSD (VOC pretrained)

Usage:
    python detect.py                     # webcam
    python detect.py --source video.mp4  # video file
    python detect.py --source image.jpg  # single image
    python detect.py --ssd-only          # skip HOG+SVM
    python detect.py --hog-only          # skip SSD
"""

import cv2
import numpy as np
import joblib
import json
import time
import argparse
from pathlib import Path
from skimage.feature import hog

# ─── Paths (adjust if needed) ───────────────────────────────────
BASE      = Path(__file__).parent
MODEL_DIR  = BASE / "models"
WEIGHT_DIR = BASE / "weights"

SVM_PATH     = MODEL_DIR / "hog_svm.pkl"
SCALER_PATH  = MODEL_DIR / "hog_scaler.pkl"
LE_PATH      = MODEL_DIR / "label_encoder.pkl"
CLASSES_PATH = MODEL_DIR / "classes.json"
PROTO_PATH   = WEIGHT_DIR / "MobileNetSSD_deploy.prototxt"
MODEL_PATH   = WEIGHT_DIR / "MobileNetSSD_deploy.caffemodel"
SSD_CLS_PATH = WEIGHT_DIR / "ssd_classes.json"

# ─── MobileNet SSD parameters ───────────────────────────────────
SSD_CONF_THRESH  = 0.40
SSD_INPUT_SIZE   = (300, 300)
SSD_MEAN         = 127.5
SSD_SCALE        = 0.007843

# ─── HOG+SVM parameters ─────────────────────────────────────────
HOG_WIN_SIZE     = (64, 64)
HOG_STRIDE       = 32         # sliding-window stride
HOG_SCALES       = [1.0, 0.75, 0.5]   # image pyramid scales
HOG_CONF_THRESH  = 1.0        # SVM decision-function threshold

# ─── Display ────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2
WINDOW_NAME = "Multi-Object Detector  |  press Q to quit"


# ════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════
def load_models(use_ssd=True, use_hog=True):
    models = {}

    if use_ssd:
        print("[INFO] Loading MobileNet SSD…")
        net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(SSD_CLS_PATH) as f:
            ssd_classes = json.load(f)
        np.random.seed(42)
        ssd_colors = np.random.randint(0, 220, (len(ssd_classes), 3), np.uint8)
        models["ssd"] = {"net": net, "classes": ssd_classes, "colors": ssd_colors}
        print(f"  ✅ SSD ready  ({len(ssd_classes)} classes)")

    if use_hog:
        print("[INFO] Loading HOG + SVM…")
        svm    = joblib.load(SVM_PATH)
        scaler = joblib.load(SCALER_PATH)
        le     = joblib.load(LE_PATH)
        with open(CLASSES_PATH) as f:
            hog_classes = json.load(f)
        np.random.seed(7)
        hog_colors = np.random.randint(100, 255, (len(hog_classes), 3), np.uint8)
        models["hog"] = {"svm": svm, "scaler": scaler, "le": le,
                         "classes": hog_classes, "colors": hog_colors}
        print(f"  ✅ HOG+SVM ready  ({len(hog_classes)} classes)")

    return models


# ════════════════════════════════════════════════════════════════
# SSD inference
# ════════════════════════════════════════════════════════════════
def detect_ssd(frame, models):
    """Run MobileNet SSD on one frame, return list of (x1,y1,x2,y2,label,conf,color)."""
    if "ssd" not in models:
        return []

    net     = models["ssd"]["net"]
    classes = models["ssd"]["classes"]
    colors  = models["ssd"]["colors"]
    h, w    = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, SSD_INPUT_SIZE),
        SSD_SCALE, SSD_INPUT_SIZE, SSD_MEAN
    )
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < SSD_CONF_THRESH:
            continue
        idx  = int(detections[0, 0, i, 1])
        box  = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        label  = classes[idx] if idx < len(classes) else "unknown"
        color  = colors[idx].tolist() if idx < len(colors) else [255, 255, 0]
        results.append((x1, y1, x2, y2, label, conf, color, "SSD"))
    return results


# ════════════════════════════════════════════════════════════════
# HOG + SVM sliding-window inference
# ════════════════════════════════════════════════════════════════
def extract_hog_feats(patch):
    patch = cv2.resize(patch, HOG_WIN_SIZE)
    gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               block_norm="L2-Hys", visualize=False)


def non_max_suppression(boxes, overlap_thresh=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = boxes[:,5].argsort()[::-1]   # sort by score
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= overlap_thresh)[0] + 1]
    return [boxes[k] for k in keep]


def detect_hog(frame, models):
    if "hog" not in models:
        return []

    svm     = models["hog"]["svm"]
    scaler  = models["hog"]["scaler"]
    le      = models["hog"]["le"]
    colors  = models["hog"]["colors"]
    h, w    = frame.shape[:2]
    wins, boxes_raw = [], []

    for scale in HOG_SCALES:
        rw, rh = int(w * scale), int(h * scale)
        if rw < HOG_WIN_SIZE[0] or rh < HOG_WIN_SIZE[1]:
            continue
        resized = cv2.resize(frame, (rw, rh))
        wy, wx  = HOG_WIN_SIZE
        for y in range(0, rh - wy, HOG_STRIDE):
            for x in range(0, rw - wx, HOG_STRIDE):
                patch = resized[y:y+wy, x:x+wx]
                feats = extract_hog_feats(patch)
                wins.append(feats)
                # Map back to original coords
                x1o = int(x / scale); y1o = int(y / scale)
                x2o = int((x+wx) / scale); y2o = int((y+wy) / scale)
                boxes_raw.append((x1o, y1o, x2o, y2o))

    if not wins:
        return []

    X_s     = scaler.transform(np.array(wins, dtype=np.float32))
    scores  = svm.decision_function(X_s)   # shape (N, n_classes)
    preds   = np.argmax(scores, axis=1)
    confs   = scores[np.arange(len(scores)), preds]

    candidates = []
    for i, (conf, pred) in enumerate(zip(confs, preds)):
        if conf < HOG_CONF_THRESH:
            continue
        x1, y1, x2, y2 = boxes_raw[i]
        candidates.append([x1, y1, x2, y2, pred, conf])

    kept = non_max_suppression(candidates)
    results = []
    for box in kept:
        x1, y1, x2, y2, pred, conf = box[:6]
        idx   = int(pred)
        label = le.inverse_transform([idx])[0]
        color = colors[idx % len(colors)].tolist()
        results.append((int(x1), int(y1), int(x2), int(y2),
                        label, float(conf), color, "HOG"))
    return results


# ════════════════════════════════════════════════════════════════
# Draw detections
# ════════════════════════════════════════════════════════════════
def draw_detections(frame, detections):
    for (x1, y1, x2, y2, label, conf, color, src) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
        tag = f"{label} {conf:.2f} [{src}]"
        (tw, th), _ = cv2.getTextSize(tag, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
    return frame


def draw_fps_info(frame, fps, n_det):
    info = f"FPS: {fps:.1f}  |  Objects: {n_det}"
    cv2.putText(frame, info, (10, 28), FONT, 0.7, (0, 255, 80), 2)


# ════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════
def run(source=0, use_ssd=True, use_hog=True):
    models = load_models(use_ssd, use_hog)

    # Open source
    src = source
    if isinstance(source, str) and source.isdigit():
        src = int(source)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # For images, just process one frame
    is_image = isinstance(source, str) and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    print(f"[INFO] Running detection  (source={source})")
    print("[INFO] Press  Q  to quit")

    fps_t = time.time()
    fps   = 0.0
    frame_count = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ─ Run detectors ─
        detections = detect_ssd(frame, models) + detect_hog(frame, models)

        # ─ Draw ─
        frame = draw_detections(frame, detections)

        # ─ FPS ─
        elapsed = time.time() - fps_t
        if elapsed > 0:
            fps = frame_count / elapsed
        draw_fps_info(frame, fps, len(detections))

        cv2.imshow(WINDOW_NAME, frame)

        if is_image:
            cv2.waitKey(0)
            break

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_count} frames @ avg {fps:.1f} FPS")


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Multi-Object Detector")
    parser.add_argument("--source",   default="0",
                        help="0=webcam | path to video/image")
    parser.add_argument("--ssd-only", action="store_true",
                        help="Run only MobileNet SSD")
    parser.add_argument("--hog-only", action="store_true",
                        help="Run only HOG+SVM")
    parser.add_argument("--conf",     type=float, default=0.40,
                        help="SSD confidence threshold (default 0.40)")
    args = parser.parse_args()

    use_ssd = not args.hog_only
    use_hog = not args.ssd_only
    SSD_CONF_THRESH = args.conf

    source = args.source
    if source.isdigit():
        source = int(source)

    run(source=source, use_ssd=use_ssd, use_hog=use_hog)