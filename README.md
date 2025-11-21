# AI Baseball Runner Advance Prediction

This project builds a complete end-to-end system that predicts whether a baserunner will advance on a sacrifice fly play using both computer vision and machine learning.

## Pipeline Overview
```
Video → YOLO Tracking → Feature Extraction → Ensemble ML Model → SAFE/OUT Prediction
```

The system processes a raw MLB game clip and outputs:
- An annotated video (bounding boxes)
- A tracker CSV (player positions)
- A features CSV (7 final model inputs)
- A prediction file with SAFE/OUT + probability

---

## Environment Setup

### 1. Create Virtual Environment

Run in Terminal:
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Go to AIBaseballproject - Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Complete Pipeline

From the project root:
```bash
cd main
python run_complete_pipeline.py --video videos/sac_fly_001.mp4 --metadata video_metadata.csv --output results
```

### Output Files

Written to `results/`:
- `*_tracker.csv` — detected players per frame
- `*_features.csv` — extracted model-ready features
- `*_prediction.txt` — SAFE/OUT + probability
- `*_annotated.mp4` — bounding box video

---

## 📁 Repository Structure
```
AIBaseballProject/
│
├── book
├── deliverables
├── main/
    ├── results/
    ├── videos/
    ├── run_complete_pipeline.py
    ├── All .csv
    └── All .py
├── GITHUB_SETUP_GUIDE
├── README.md
└── Requirements.txt

```

---

## 📘 Full Project Documentation

A full Quarto book documenting all 4 weeks of development is available in:
```
/book/
```
