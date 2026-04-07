# Real-Time Multi-Object Detection
## HOG+SVM (MS-COCO subset) + MobileNet SSD

### Project Structure
```
detector/
├── detect.py            ← main script (run this)
├── requirements.txt
├── models/
│   ├── hog_svm.pkl          HOG + LinearSVM classifier
│   ├── hog_scaler.pkl       StandardScaler for HOG features
│   ├── label_encoder.pkl    sklearn LabelEncoder
│   └── classes.json         HOG class names
└── weights/
    ├── MobileNetSSD_deploy.prototxt
    ├── MobileNetSSD_deploy.caffemodel
    └── ssd_classes.json     SSD class names (VOC)
```

### Setup (VS Code / local)
```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with webcam
python detect.py

# 4. Run on a video file
python detect.py --source myvideo.mp4

# 5. Run on an image
python detect.py --source photo.jpg

# 6. SSD only (faster, higher accuracy)
python detect.py --ssd-only

# 7. HOG+SVM only
python detect.py --hog-only

# 8. Adjust confidence
python detect.py --conf 0.5
```

### Detectors
| Detector | Classes | Speed | Notes |
|---|---|---|---|
| MobileNet SSD | 20 (VOC) | ~30 FPS | Deep learning, high accuracy |
| HOG + LinearSVM | 8 (COCO subset) | ~5-10 FPS | Classical CV, sliding window |

### Classes Detected
**SSD (MobileNet):** aeroplane, bicycle, bird, boat, bottle, bus, car, cat,
chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor

**HOG+SVM (COCO trained):** person, car, cat, dog, bottle, chair, cell phone, book

### Controls
- Press **Q** or **ESC** to quit
