# AI Baseball Runner Advance Prediction

This project builds a complete end-to-end system that predicts whether a baserunner will advance on a sacrifice fly play using both computer vision and machine learning.

**Pipeline:**
> Video → YOLO Tracking → Feature Extraction → Ensemble ML Model → SAFE/OUT Prediction

The system processes a raw MLB game clip and outputs:
- an annotated video (bounding boxes)
- a tracker CSV (player positions)
- a features CSV (7 final model inputs)
- a prediction file with SAFE/OUT + probability

---

## Environment Setup

### 1. Create virtual environment
Run in Terminal
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

2. Install dependencies

pip install -r requirements.txt

▶️ Running the Complete Pipeline

From the project root:

cd deliverables/keaton/deliverable_4

python run_complete_pipeline.py \
  --video ../../../videos/sac_fly_001.mp4 \
  --metadata data/video_metadata.csv \
  --output ../../results

Output files (written to results/):

    *_tracker.csv — detected players per frame

    *_features.csv — extracted model-ready features

    *_prediction.txt — SAFE/OUT + probability

    *_annotated.mp4 — bounding box video

📁 Repository Structure

AIBaseballProject/
│
├── videos/
├── results/
├── requirements.txt
├── README.md
│
└── deliverables/
    └── keaton/
        └── deliverable_4/
            ├── run_complete_pipeline.py
            ├── src/
            └── data/

📘 Full Project Documentation

A full Quarto book documenting all 4 weeks of development is available in:
/book/
