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
Please go to Professor_Setup_Guide.md 

### Output Files

Written to `results/`:
- `*_tracker.csv` — detected players per frame
- `*_features.csv` — extracted model-ready features
- `*_prediction.txt` — SAFE/OUT + probability
- `*_annotated.mp4` — bounding box video


---

## Full Project Documentation

Full project documentation with weekly progress is available in the /book directory as a Quarto book.
To build and view the documentation:

```bash
cd book
quarto render
```

