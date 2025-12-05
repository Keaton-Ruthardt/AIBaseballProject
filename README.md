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

### Environnment Setup
Check the Environmnet Setup Folder 

### Output Files

Written to `results/`:
- `*_tracker.csv` — detected players per frame
- `*_features.csv` — extracted model-ready features
- `*_prediction.txt` — SAFE/OUT + probability
- `*_annotated.mp4` — bounding box video

---

## Repository Structure
```
AIBaseballProject/
│
├── main/                       # Main source code and data
│   ├── run_complete_pipeline.py      # Main pipeline script
│   ├── simple_tracker.py             # YOLO-based player tracking
│   ├── pixel_to_feet_converter.py    # Coordinate conversion
│   ├── feature_preparation.py        # Feature extraction
│   ├── original_ensemble_model_D3.py # ML ensemble model
│   ├── validator.py                  # Input validation
│   │
│   ├── videos/                       # Sample video files
│   │   ├── sac_fly_001.mp4
│   │   └── ...
│   │
│   ├── train_data.csv                # Training dataset
│   ├── test_data.csv                 # Test dataset
│   ├── glossary.csv                  # Feature glossary
│   ├── outfield_position.csv         # Position tracking data
│   ├── video_metadata.csv            # Video metadata
│   └── yolov8n.pt                    # YOLO model weights
│
├── book/                       # Quarto documentation
│   ├── _quarto.yml
│   ├── index.qmd
│   ├── week1.qmd
│   ├── week2.qmd
│   ├── week3.qmd
│   └── week4.qmd
│
├── deliverables/               # Weekly team deliverables
│   ├── keaton/
│   ├── joshua/
│   ├── diego/
│   ├── duoduo/
│   └── samuel/
│
├── results/                    # Output directory (generated)
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore

```

---

## Full Project Documentation

Full project documentation with weekly progress is available in the /book directory as a Quarto book.
To build and view the documentation:

```bash
cd book
quarto render
```

