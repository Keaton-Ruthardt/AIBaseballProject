# Baseball Runner Advance Prediction Pipeline

Complete end-to-end pipeline: Video → Tracking → Features → ML Prediction

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv_deliverable4
venv_deliverable4\Scripts\activate  # Windows
# source venv_deliverable4/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python run_complete_pipeline.py --video your_video.mp4 --metadata video_metadata.csv --output results
```

## What You Get

- **Exact ML predictions** (not heuristics!)
- **Annotated videos** with player detection boxes
- **Feature extraction** from video + Statcast
- **79-90% prediction accuracy** from trained ensemble model

## Input Format

**video_metadata.csv** (one file for all videos):
```csv
video_filename,video_id,play_outcome,runner_base,exit_velocity,launch_angle,hit_distance,hangtime,...
sac_fly_001.mp4,sac_fly_001,safe,3B,109.8,16,305,3,...
sac_fly_002.mp4,sac_fly_002,safe,3B,78.6,39,273,3,...
```

The pipeline automatically extracts the row matching your video filename!

## Output Files

1. `{video}_tracker.csv` - Player positions per frame
2. `{video}_annotated.mp4` - Video with bounding boxes
3. `{video}_features.csv` - All 8 extracted features
4. `{video}_prediction.txt` - Final prediction + probability

## Example Result

```
Video: sac_fly_001
Prediction: RUNNER ADVANCES (SAFE)
Probability: 0.791881
Confidence: 79.2%
```

See **SETUP_INSTRUCTIONS.txt** for detailed setup and troubleshooting.

## Model Details

- **Ensemble:** Random Forest + Gradient Boosting + Logistic Regression
- **Training data:** 15,533 sacrifice fly plays
- **Features:** 7 total (6 numerical, 1 categorical fielder position)
- **Performance:** AUC 0.90, Log Loss 0.40

## Package Contents

- ✅ Main pipeline script
- ✅ All source code (tracker, converter, model)
- ✅ Training data (train_data.csv, test_data.csv)
- ✅ Requirements.txt
- ✅ Complete setup instructions
