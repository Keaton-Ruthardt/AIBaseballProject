# Complete Setup Guide for Professor
## Baseball Runner Advancement Prediction System

**Course:** Artificial Intelligence 
**Date:** December 2025

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Instructions](#installation-instructions)
   - [Windows Setup](#windows-setup)
   - [macOS Setup](#macos-setup)
   - [Linux Setup](#linux-setup)
3. [Verify Installation](#verify-installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Viewing Demo Videos](#viewing-demo-videos)
6. [Understanding the Outputs](#understanding-the-outputs)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

**Don't have Python?** Continue reading for full setup instructions.

---

## System Requirements

### **Minimum Requirements**

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+) |
| **Python Version** | Python 3.9, 3.10, or 3.11 (3.11 recommended) |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Disk Space** | 5 GB free space (for project + dependencies) |
| **Video Player** | VLC Media Player or any MP4-compatible player |

### **Optional (For Better Performance)**

| Component | Benefit |
|-----------|---------|
| **GPU** | NVIDIA GPU with CUDA support (10x faster processing) |
| **SSD** | Faster video processing |

**Note:** The system works perfectly fine on CPU-only machines. GPU is optional for speed improvements.

---

## Installation Instructions

Choose your operating system:

---

## Windows Setup

### **Step 1: Install Python 3.11**

#### Option A: Download from Python.org (Recommended)

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11.x** (latest stable version)
3. Run the installer
4. **IMPORTANT:** Check the box that says "Add Python to PATH"
5. Click "Install Now"

#### Option B: Use Microsoft Store

1. Open Microsoft Store
2. Search for "Python 3.11"
3. Click "Get" to install

#### Verify Python Installation

Open Command Prompt (cmd) and run:

```cmd
python --version
```

You should see: `Python 3.11.x`

If you see an error or wrong version:
- Try `python3 --version` instead
- Or `py --version`

### **Step 2: Project Files**

1. Every file you need will be inside of MainCodeFolder:
    ```
    Download all files just like in the Github
    ```

2. Extract to a location like:
   ```
   C:\Users\YourName\Desktop\Baseball_Project\
   ```

3. Open Command Prompt in this folder:
   - **Method 1:** Shift + Right-click folder → "Open PowerShell window here" or "Open command window here"
   - **Method 2:** Open Command Prompt, then:
     ```cmd
     cd C:\Users\YourName\Desktop\Baseball_Project
     ```

### **Step 3: Create Virtual Environment (Recommended)**

This keeps dependencies isolated and prevents conflicts.

```cmd
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) at the start of your command prompt
```

**Note:** If you get an execution policy error, run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Step 4: Install Dependencies**

```cmd
# Make sure you're in the project folder with requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `opencv-python` (Computer vision)
- `ultralytics` (YOLOv8 player detection)
- `scikit-learn` (Machine learning)
- `pandas` & `numpy` (Data processing)
- `matplotlib` (Visualization)

**Installation takes 5-10 minutes** depending on your internet speed.

### **Step 5: Verify Installation**

```cmd
# Test Python packages
python -c "import cv2, torch, ultralytics, sklearn, pandas; print('All packages installed successfully!')"
```

If you see "All packages installed successfully!" - you're ready!

---

## macOS Setup

### **Step 1: Install Python 3.11**

#### Check if Python 3 is already installed:

```bash
python3 --version
```

If you see Python 3.9, 3.10, or 3.11 - you're good! Skip to Step 2.

#### Install Python 3.11 using Homebrew:

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python:
   ```bash
   brew install python@3.11
   ```

3. Verify:
   ```bash
   python3 --version
   ```

#### Alternative: Download from Python.org

1. Go to https://www.python.org/downloads/macos/
2. Download Python 3.11.x macOS installer
3. Run the .pkg file and follow instructions

### **Step 2: Extract Project Files**

1. Double-click the ZIP file to extract
2. Move folder to convenient location (e.g., Desktop or Documents)
3. Open Terminal and navigate to project:
   ```bash
   cd ~/Desktop/Baseball_Project
   ```

### **Step 3: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) at the start of your prompt
```

### **Step 4: Install Dependencies**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**macOS-specific note:** If you get an error about missing command line tools:
```bash
xcode-select --install
```

### **Step 5: Verify Installation**

```bash
python -c "import cv2, torch, ultralytics, sklearn, pandas; print('All packages installed successfully!')"
```

---

## Linux Setup

### **Step 1: Install Python 3.11**

#### Ubuntu/Debian:

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# Verify
python3.11 --version
```

#### Fedora/RHEL:

```bash
# Install Python 3.11
sudo dnf install python3.11

# Verify
python3.11 --version
```

#### Arch Linux:

```bash
# Install Python
sudo pacman -S python

# Verify
python --version
```

### **Step 2: Extract Project Files**

```bash
# Extract ZIP
unzip Baseball_Prediction_Project_Keaton.zip

# Navigate to folder
cd Baseball_Prediction_Project_Keaton
```

### **Step 3: Create Virtual Environment**

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate
```

### **Step 4: Install System Dependencies**

Some packages need system libraries:

#### Ubuntu/Debian:

```bash
sudo apt install -y \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev
```

#### Fedora:

```bash
sudo dnf install -y \
    python3-devel \
    mesa-libGL \
    glib2
```

### **Step 5: Install Python Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **Step 6: Verify Installation**

```bash
python -c "import cv2, torch, ultralytics, sklearn, pandas; print('All packages installed successfully!')"
```

---

## Verify Installation

### **Quick Verification Test**

Run this command to verify everything is working:

```bash
python -c "
import sys
print(f'Python version: {sys.version}')

import cv2
print(f'OpenCV version: {cv2.__version__}')

import torch
print(f'PyTorch version: {torch.__version__}')

from ultralytics import YOLO
print('YOLOv8 imported successfully')

import sklearn
print(f'scikit-learn version: {sklearn.__version__}')

import pandas
print(f'pandas version: {pandas.__version__}')

print('\n✅ All dependencies installed correctly!')
"
```

**Expected Output:**
```
Python version: 3.11.x
OpenCV version: 4.x.x
PyTorch version: 2.x.x
YOLOv8 imported successfully
scikit-learn version: 1.x.x
pandas version: 2.x.x

✅ All dependencies installed correctly!
```

---

## Running the Pipeline

### **What Does the Pipeline Do?**

The pipeline processes baseball videos in 3 stages:
1. **Video Tracking** - Detects and tracks outfielders in the video
2. **Feature Extraction** - Extracts baseball metrics (exit velocity, launch angle, etc.)
3. **ML Prediction** - Predicts if the runner will advance (SAFE) or stay (OUT)

### **Basic Usage**

#### Test on Pre-Provided Video:

```bash
python run_complete_pipeline.py \
    --video Video-MetaData/sac_fly_002.mp4 \
    --metadata Video-MetaData/video_metadata.csv \
    --output test_output
```

#### What You'll See:

```
================================================================================
COMPLETE PIPELINE: sac_fly_002
================================================================================

Inputs:
  Video:    DataBatch2-20251112T182025Z-1-001/DataBatch2/sac_fly_002.mp4
  Metadata: DataBatch2-20251112T182025Z-1-001/DataBatch2/video_metadata.csv

================================================================================
STAGE 1: VIDEO TRACKING
================================================================================

[1/3] Initializing tracker...
   [OK] Tracker initialized

[2/3] Processing video: DataBatch2-20251112T182025Z-1-001/DataBatch2/sac_fly_002.mp4
   Processing frames 180-400 (outfield action)
   Processed 220 frames
   [OK] Annotated video saved: test_output/sac_fly_002_annotated.mp4

[3/3] Saving tracker CSV...
   [OK] Tracker CSV saved: test_output/sac_fly_002_tracker.csv

================================================================================
STAGE 2: FEATURE EXTRACTION
================================================================================

[1/4] Initializing pixel-to-feet converter...
   [OK] Converter initialized

[2/4] Loading tracker data...
   [OK] Loaded 220 frames
   [OK] Catch detected at frame 195

[3/4] Calculating video-derived features...
   Player distance: 52.3 feet (to nearest outfielder)
   Fielder: CF

[4/4] Loading Statcast data...
   [OK] Loaded Statcast metrics

   [OK] All features extracted successfully

   [OK] Features saved: test_output/sac_fly_002_features.csv

================================================================================
STAGE 3: MODEL PREDICTION
================================================================================

[1/4] Preparing features for model...

   Input Features:
     runner_base         : 3
     exit_speed          : 98.5
     launch_angle        : 28.0
     hit_distance        : 385.0
     hangtime            : 5.2
     fielder_pos         : CF
     ...

[2/4] Loading pre-trained ensemble model...
   [OK] Pre-trained model loaded from disk (10x faster!)

[3/4] Preparing video features for prediction...
   [OK] Feature vector prepared: (1, 10)

[4/4] Making prediction with trained ensemble model...
   [OK] Prediction complete
   Model output: 0.723456 (raw probability)

================================================================================
FINAL RESULTS
================================================================================

Video: sac_fly_002
Prediction: RUNNER ADVANCES (SAFE)
Probability: 0.723456
Confidence: 72.3%

[OK] Results saved: test_output/sac_fly_002_prediction.txt

================================================================================
PIPELINE COMPLETE [OK]
================================================================================
```

**Processing Time:** 2-5 minutes on CPU, 30 seconds on GPU

### **View the Results**

After the pipeline completes, check the `test_output/` folder:

```bash
# List output files
ls test_output/

# You should see:
# sac_fly_002_annotated.mp4      - Video with player detections
# sac_fly_002_tracker.csv        - Frame-by-frame tracking data
# sac_fly_002_features.csv       - Extracted ML features
# sac_fly_002_prediction.txt     - Final prediction result
```

### **Run on Different Video**

Try another video from the dataset:

```bash
python run_complete_pipeline.py \
    --video DataBatch2-20251112T182025Z-1-001/DataBatch2/sac_fly_003.mp4 \
    --metadata DataBatch2-20251112T182025Z-1-001/DataBatch2/video_metadata.csv \
    --output test_output_sac3
```

---

## Viewing Demo Videos

### **What Are Demo Videos?**

Demo videos show side-by-side comparison:
- **Left side:** Original broadcast video
- **Right side:** Video with player detection bounding boxes
- **Bottom overlay:** ML model prediction (SAFE/OUT with probability)

### **Pre-Generated Demo Videos**

We've included several ready-to-watch demo videos in `demo_output/`:

1. **`sac_fly_002_CLASS_DEMO.mp4`** ⭐ **PRIMARY DEMO** (81 MB)
   - Best example showcasing the system
   - Clear player detection
   - High-confidence prediction

2. **`sac_fly_003_CLASS_DEMO.mp4`** (114 MB)
   - Shows system handling different camera angles

3. **`sacfly057_sac_fly_042_CLASS_DEMO.mp4`** (61 MB)
   - Different game scenario

4. **`sacfly057_sac_fly_043_CLASS_DEMO.mp4`** (119 MB)
   - Another example

### **How to Watch Demo Videos**

#### **Windows:**
1. Navigate to `demo_output/` folder
2. Double-click `sac_fly_002_CLASS_DEMO.mp4`
3. Video opens in Windows Media Player or default video player

#### **macOS:**
1. Navigate to `demo_output/` folder
2. Double-click `sac_fly_002_CLASS_DEMO.mp4`
3. Video opens in QuickTime Player

#### **Linux:**
```bash
# Using VLC (install if needed: sudo apt install vlc)
vlc demo_output/sac_fly_002_CLASS_DEMO.mp4

# Or mpv (install if needed: sudo apt install mpv)
mpv demo_output/sac_fly_002_CLASS_DEMO.mp4
```

### **Create Your Own Demo Video**

Want to create a demo video for a different game?

```bash
# This script runs the pipeline AND creates a demo video
python create_demo_video.py
```

**Note:** By default, this processes `sac_fly_002`. To change which video:

1. Open `create_demo_video.py` in a text editor
2. Find line 114: `video_name = "sac_fly_002"`
3. Change to: `video_name = "sac_fly_003"` (or any other video)
4. Update line 115 with the correct video path
5. Save and run: `python create_demo_video.py`

---

## Understanding the Outputs

### **Output Files Explained**

For each video processed, the system creates 4 files:

#### **1. Annotated Video (`*_annotated.mp4`)**

**What it is:** The original video with colored bounding boxes around detected players.

**Bounding Box Colors:**
- **Blue boxes** = Left Fielder (LF)
- **Green boxes** = Center Fielder (CF)
- **Red boxes** = Right Fielder (RF)
- **Yellow box** = Baseball (when visible)

**How to view:** Double-click to open in any video player

**Example:** `test_output/sac_fly_002_annotated.mp4`

---

#### **2. Tracker CSV (`*_tracker.csv`)**

**What it is:** Frame-by-frame tracking data with player positions.

**Columns:**
- `frame_number` - Frame index (0 to 220)
- `num_outfielders_detected` - How many players detected (0-3)
- `player_0_x`, `player_0_y` - First detected player position (pixels)
- `player_1_x`, `player_1_y` - Second player position
- `player_2_x`, `player_2_y` - Third player position
- `catcher_player_id` - Which player caught the ball (-1 = none)
- `ball_detected` - Whether baseball was visible (True/False)

**How to view:**
```bash
# View in terminal
head test_output/sac_fly_002_tracker.csv

# Or open in Excel/Google Sheets
```

**Example snippet:**
```csv
frame_number,num_outfielders_detected,player_0_x,player_0_y,player_1_x,player_1_y,player_2_x,player_2_y,catcher_player_id,ball_detected
0,3,91.0,169.0,576.0,155.0,1205.0,164.0,-1,False
1,3,90.0,168.0,576.0,155.0,1206.0,167.0,-1,True
2,3,90.0,169.0,577.0,155.0,1208.0,168.0,-1,False
...
195,3,88.0,171.0,578.0,156.0,1210.0,169.0,1,True  <-- Catch detected! (player_1 caught it)
```

---

#### **3. Features CSV (`*_features.csv`)**

**What it is:** Extracted features used by the ML model for prediction.

**Columns:**
- `video_name` - Video identifier
- `runner_base` - Which base the runner started on (3 = 3rd base)
- `exit_speed` - Ball exit velocity in mph (from Statcast)
- `launch_angle` - Ball launch angle in degrees (from Statcast)
- `hit_distance` - Ball travel distance in feet (from Statcast)
- `hangtime` - Time ball was in the air in seconds (from Statcast)
- `fielder_pos` - Which fielder caught the ball (LF/CF/RF)
- `exit_speed_squared` - exit_speed² (engineered feature)
- `launch_direction` - Ball direction (default 0)
- `player_distance_ft` - Distance between fielders in feet

**How to view:**
```bash
cat test_output/sac_fly_002_features.csv
```

**Example:**
```csv
video_name,runner_base,exit_speed,launch_angle,hit_distance,hangtime,fielder_pos,exit_speed_squared,launch_direction,player_distance_ft
sac_fly_002,3,98.5,28.0,385.0,5.2,CF,9702.25,0,52.3
```

**This single row contains all 10 features the model uses to make a prediction.**

---

#### **4. Prediction Text File (`*_prediction.txt`)**

**What it is:** Human-readable prediction result.

**How to view:**
```bash
cat test_output/sac_fly_002_prediction.txt
```

**Example:**
```
Video: sac_fly_002
Prediction: ADVANCES
Probability: 0.723

Features:
  runner_base: 3
  exit_speed: 98.5
  launch_angle: 28.0
  hit_distance: 385.0
  hangtime: 5.2
  fielder_pos: CF
  exit_speed_squared: 9702.25
  launch_direction: 0
  player_distance_ft: 52.3
```

**Interpretation:**
- **Prediction: ADVANCES** = Model predicts runner will be SAFE (score)
- **Probability: 0.723** = 72.3% confidence in this prediction
- If probability > 0.5 → Predict ADVANCES (SAFE)
- If probability < 0.5 → Predict STAYS (OUT)

---

### **Sample Outputs Included**

We've included pre-generated outputs for 20+ videos in the `demo_output/` folder so you can examine results without running the pipeline:

**Example videos with complete outputs:**
- `sac_fly_001` - All 4 output files
- `sac_fly_002` - All 4 output files + CLASS_DEMO video
- `sac_fly_003` - All 4 output files + CLASS_DEMO video
- `sac_fly_005` through `sac_fly_015` - All outputs
- And more...

**Total:** ~80 files showing results from 20 processed videos

---

## Troubleshooting

### **Common Issues and Solutions**

#### **Issue 1: "python: command not found"**

**Solution:**
- **Windows:** Try `py` or `python3` instead of `python`
- **macOS/Linux:** Use `python3` instead of `python`
- Verify Python is in PATH: Reinstall Python with "Add to PATH" checked

---

#### **Issue 2: "No module named 'cv2'" or similar**

**Problem:** Dependencies not installed

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

#### **Issue 3: Pipeline runs slowly (5+ minutes)**

**Normal on CPU!** Video processing is compute-intensive.

**Solutions to speed up:**
- Use a smaller video (try `sac_fly_001.mp4` which is shorter)
- Use pre-trained model (included: `trained_model.pkl`)
- Use GPU if available (automatic detection)

**Note:** First run is slower because YOLOv8 downloads model weights (~6MB). Subsequent runs are faster.

---

#### **Issue 4: "FileNotFoundError: video_metadata.csv"**

**Problem:** Metadata file path is incorrect

**Solution:**
Make sure you're using the correct path with forward slashes `/` or backslashes `\\`:

**Windows:**
```cmd
python run_complete_pipeline.py --video DataBatch2-20251112T182025Z-1-001\DataBatch2\sac_fly_002.mp4 --metadata DataBatch2-20251112T182025Z-1-001\DataBatch2\video_metadata.csv --output test_output
```

**macOS/Linux:**
```bash
python run_complete_pipeline.py --video DataBatch2-20251112T182025Z-1-001/DataBatch2/sac_fly_002.mp4 --metadata DataBatch2-20251112T182025Z-1-001/DataBatch2/video_metadata.csv --output test_output
```

---

#### **Issue 5: Demo video won't play**

**Problem:** Missing video codec

**Solution:**
- **Install VLC Media Player:** https://www.videolan.org/ (free, plays everything)
- Or install codecs:
  - Windows: K-Lite Codec Pack
  - macOS: Should work in QuickTime
  - Linux: `sudo apt install ubuntu-restricted-extras`

---

#### **Issue 6: "RuntimeError: CUDA not available" or GPU warnings**

**This is normal!** GPU is optional.

The system works perfectly on CPU. GPU just makes it faster.

**To use GPU (optional):**
1. Install NVIDIA drivers
2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

**Note:** CPU-only is fine for grading/demonstration purposes.

---

#### **Issue 7: Virtual environment activation doesn't work**

**Windows PowerShell error:**
```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
venv\Scripts\activate
```

**Alternative:** Use Command Prompt (cmd) instead of PowerShell

---

#### **Issue 8: Permission denied errors (macOS/Linux)**

**Solution:**
```bash
# Make scripts executable
chmod +x run_complete_pipeline.py
chmod +x create_demo_video.py

# Or run with python explicitly
python run_complete_pipeline.py ...
```

---

#### **Issue 9: Out of memory errors**

**Problem:** System has less than 8GB RAM

**Solution:**
- Close other applications
- Process one video at a time
- Use shorter videos from dataset

---

#### **Issue 10: "trained_model.pkl not found" warning**

**This is OK!** The system will train a model from scratch (takes 5-10 minutes once).

**To speed up future runs:**
```bash
# Train and save model once
python train_and_save_model.py

# Future runs will be 10x faster using trained_model.pkl
```

**Note:** We've included `trained_model.pkl` in the submission, so you shouldn't see this warning.

---

## Advanced Usage

### **Customizing Frame Range**

By default, the tracker processes frames 180-400 (where outfield action happens).

To change this, edit `run_complete_pipeline.py` line 43:

```python
# Original:
results = tracker.process_video(video_path, output_path=output_video, start_frame=180, end_frame=400)

# Process entire video:
results = tracker.process_video(video_path, output_path=output_video, start_frame=0, end_frame=None)

# Custom range (frames 200-300):
results = tracker.process_video(video_path, output_path=output_video, start_frame=200, end_frame=300)
```

---


### **Understanding Model Performance**

The ensemble model uses:
1. **Random Forest** (40% weight) - Handles non-linear patterns
2. **Gradient Boosting** (40% weight) - Handles sequential patterns
3. **Logistic Regression** (20% weight) - Baseline linear model

---

### **File Structure Reference**

```
Baseball_Prediction_Project_Keaton/
│
├── run_complete_pipeline.py          # ⭐ MAIN SCRIPT - Run this
├── create_demo_video.py              # Create demo videos
├── DuoDuo.py                         # Validation metrics
├── original_ensemble_model.py        # ML model (45 features)
├── requirements.txt                  # Python dependencies
├── trained_model.pkl                 # Pre-trained model (13 MB)
│
├── demo_output/                      # Pre-generated outputs
│   ├── sac_fly_002_CLASS_DEMO.mp4   # ⭐ PRIMARY DEMO VIDEO
│   ├── sac_fly_002_annotated.mp4    # Video with detections
│   ├── sac_fly_002_tracker.csv      # Tracking data
│   ├── sac_fly_002_features.csv     # ML features
│   ├── sac_fly_002_prediction.txt   # Prediction result
│   └── ... (20+ more videos)
│
├── DataBatch2-20251112T182025Z-1-001/
│   └── DataBatch2/
│       ├── video_metadata.csv        # ⭐ REQUIRED - Statcast data
│       ├── sac_fly_001.mp4           # Test videos
│       ├── sac_fly_002.mp4           # Test videos
│       ├── sac_fly_003.mp4           # Test videos
│       └── ... (50+ videos)
│
├── deliverables/keaton/deliverable_4/src/
│   ├── simple_tracker.py             # Player detection (YOLOv8)
│   ├── pixel_to_feet_converter.py   # Coordinate transformation
│   ├── original_ensemble_model_D3.py # ML model (10 features)
│   └── feature_preparation.py        # Feature engineering
│
├── train_data.csv                    # ML training data
├── test_data.csv                     # ML test data
├── outfield_position.csv             # Position tracking data
│
├── PROFESSOR_SETUP_GUIDE.md          # ⭐ THIS FILE
├── COMPLETE_PIPELINE_README.md       # Technical documentation
├── FINAL_SOLUTION_GUIDE.md           # Solution overview
├── Project_Summary.md                # Project context
└── ... (additional documentation)
```

---

## Quick Reference Commands

### **Essential Commands**

```bash
# 1. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies (one time)
pip install -r requirements.txt

# 3. Run pipeline on a video
python run_complete_pipeline.py --video DataBatch2-20251112T182025Z-1-001/DataBatch2/sac_fly_002.mp4 --metadata DataBatch2-20251112T182025Z-1-001/DataBatch2/video_metadata.csv --output test_output

# 4. Create demo video
python create_demo_video.py

# 5. View results
# Windows:
explorer test_output
# macOS:
open test_output
# Linux:
xdg-open test_output

# 6. Deactivate virtual environment when done
deactivate
```

---

## Summary Checklist

**Before starting:**
- [ ] Python 3.9+ installed
- [ ] Project ZIP extracted
- [ ] Terminal/Command Prompt open in project folder

**Setup:**
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Installation verified (no errors)

**Running:**
- [ ] Watch demo video: `demo_output/sac_fly_002_CLASS_DEMO.mp4`
- [ ] Run pipeline: `python run_complete_pipeline.py ...`
- [ ] Check outputs in `test_output/` folder
- [ ] View prediction result: `test_output/sac_fly_002_prediction.txt`

**Expected Results:**
- [ ] Pipeline completes without errors
- [ ] 4 output files generated for each video
- [ ] Prediction probability between 0-1
- [ ] Demo video plays correctly

---

## Getting Help

**If you encounter issues:**

1. **Check this troubleshooting section** (most common issues covered)
2. **Check the log output** - errors usually explain the problem
3. **Verify file paths** - ensure videos and metadata paths are correct
4. **Check Python version** - must be 3.9, 3.10, or 3.11

**For questions:**
- Email: keatonruthardt10@gmail.com
- See additional documentation in README files

---

## System Architecture Overview

**For understanding how it works:**

```
INPUT VIDEO (sac_fly_002.mp4)
          |
          v
┌─────────────────────────────────────┐
│  STAGE 1: VIDEO TRACKING            │
│  - Detect players using YOLOv8      │
│  - Track outfielders frame-by-frame │
│  - Identify who caught the ball     │
└─────────────┬───────────────────────┘
              v
    OUTPUT: tracker.csv (220 frames of data)
              |
              v
┌─────────────────────────────────────┐
│  STAGE 2: FEATURE EXTRACTION        │
│  - Load Statcast data (exit velo,   │
│    launch angle, distance, etc.)    │
│  - Convert pixel coords to feet     │
│  - Determine fielder position       │
└─────────────┬───────────────────────┘
              v
    OUTPUT: features.csv (10 features)
              |
              v
┌─────────────────────────────────────┐
│  STAGE 3: ML PREDICTION             │
│  - Load pre-trained ensemble model  │
│  - Random Forest + Gradient Boost + │
│    Logistic Regression              │
│  - Predict probability (0-1)        │
└─────────────┬───────────────────────┘
              v
    OUTPUT: prediction.txt
    "ADVANCES" (72.3% confidence)
```
