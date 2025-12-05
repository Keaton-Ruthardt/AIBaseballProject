"""
Complete End-to-End Pipeline
Single command: Video → Tracker → Features → Model Prediction

Usage:
    python run_complete_pipeline.py --video VIDEO.mp4 --statcast STATCAST.csv
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'deliverables/keaton/deliverable_4/src')

import pandas as pd
import numpy as np
from pixel_to_feet_converter import BaseballFieldConverter
from original_ensemble_model_D3 import RunnerAdvancePredictor
from simple_tracker import SimpleOutfielderTracker


def run_tracker(video_path, output_csv, output_video=None):
    """
    Run tracker on video to detect players and generate tracker CSV.

    Args:
        video_path: Path to video file
        output_csv: Path to save tracker CSV
        output_video: Path to save annotated video (optional)

    Returns:
        Path to tracker CSV
    """
    print("\n" + "="*80)
    print("STAGE 1: VIDEO TRACKING")
    print("="*80)

    print(f"\n[1/3] Initializing tracker...")
    tracker = SimpleOutfielderTracker()
    print("   [OK] Tracker initialized")

    print(f"\n[2/3] Processing video: {video_path}")
    print("   Processing frames 180-300 (outfield action)")
    results = tracker.process_video(video_path, output_path=output_video, start_frame=180, end_frame=300)
    print(f"   Processed {len(results)} frames")
    if output_video:
        print(f"   [OK] Annotated video saved: {output_video}")

    print(f"\n[3/3] Saving tracker CSV...")
    tracker.export_simple_csv(results, output_csv)
    print(f"   [OK] Tracker CSV saved: {output_csv}")

    return output_csv


def extract_features(tracker_csv, statcast_csv, video_name):
    """
    Extract all features from tracker and statcast data.

    Args:
        tracker_csv: Path to tracker CSV file
        statcast_csv: Path to statcast CSV file
        video_name: Name of video (for output)

    Returns:
        DataFrame with all features
    """
    print("\n" + "="*80)
    print("STAGE 2: FEATURE EXTRACTION")
    print("="*80)

    # Initialize converter
    print("\n[1/4] Initializing pixel-to-feet converter...")
    converter = BaseballFieldConverter(video_width=1280, video_height=720)
    print("   [OK] Converter initialized")

    # Load tracker data
    print("\n[2/4] Loading tracker data...")
    tracker_df = pd.read_csv(tracker_csv)
    print(f"   [OK] Loaded {len(tracker_df)} frames")

    # Find catch frame
    catch_frames = tracker_df[tracker_df['catcher_player_id'] >= 0]
    if len(catch_frames) > 0:
        catch_frame = catch_frames.iloc[0]
        print(f"   [OK] Catch detected at frame {catch_frame['frame_number']}")
    else:
        print("   ! No catch detected, using last detection")
        detections = tracker_df[tracker_df['num_outfielders_detected'] > 0]
        if len(detections) > 0:
            catch_frame = detections.iloc[-1]
        else:
            raise ValueError("No detections found in tracker CSV!")

    # Extract catch position and calculate features
    print("\n[3/4] Calculating video-derived features...")
    catch_x_px = catch_frame['player_0_x']
    catch_y_px = catch_frame['player_0_y']

    # Get fielder position
    if pd.notna(catch_x_px) and pd.notna(catch_y_px):
        fielder_pos = converter.get_fielder_position_label(catch_x_px, catch_y_px)
    else:
        fielder_pos = 'CF'  # Default to center field

    # Calculate player distance (distance between outfielders)
    player_distance = None

    # Find all detected players at catch frame
    catcher_id = catch_frame.get('catcher_player_id', 0)
    catcher_x = catch_frame.get(f'player_{catcher_id}_x')
    catcher_y = catch_frame.get(f'player_{catcher_id}_y')

    if pd.notna(catcher_x) and pd.notna(catcher_y):
        # Look for other nearby players
        min_distance_px = float('inf')

        for i in range(20):  # Check up to 20 players
            if i == catcher_id:
                continue  # Skip the catcher

            player_x = catch_frame.get(f'player_{i}_x')
            player_y = catch_frame.get(f'player_{i}_y')

            if pd.notna(player_x) and pd.notna(player_y):
                # Calculate distance
                distance_px = np.sqrt((player_x - catcher_x)**2 + (player_y - catcher_y)**2)
                if distance_px < min_distance_px:
                    min_distance_px = distance_px

        # Convert to feet if we found another player
        if min_distance_px != float('inf'):
            player_distance = min_distance_px / converter.pixels_per_foot_horizontal
            print(f"   Player distance: {player_distance:.1f} feet (to nearest outfielder)")

    print(f"   Fielder: {fielder_pos}")

    # Load Statcast data
    print("\n[4/4] Loading Statcast data...")
    statcast_df = pd.read_csv(statcast_csv)

    # Handle runner_base conversion (3B -> 3)
    runner_base = statcast_df['runner_base'].iloc[0]
    if isinstance(runner_base, str):
        runner_base = int(runner_base.replace('B', ''))

    print(f"   [OK] Loaded Statcast metrics")

    # Combine all features
    # Note: launch_direction included as 0 for model compatibility (model was trained with it)
    features = {
        'video_name': video_name,
        'runner_base': runner_base,
        'exit_speed': statcast_df['exit_speed'].iloc[0],
        'launch_angle': statcast_df['launch_angle'].iloc[0],
        'hit_distance': statcast_df['hit_distance'].iloc[0],
        'hangtime': statcast_df['hangtime'].iloc[0],
        'fielder_pos': fielder_pos,
        'exit_speed_squared': statcast_df['exit_speed'].iloc[0] ** 2,
        'launch_direction': 0,  # Default value for model compatibility
        'player_distance_ft': player_distance if player_distance is not None else np.nan
    }

    print("\n   [OK] All features extracted successfully")

    return pd.DataFrame([features])


def run_model_prediction(features_df):
    """
    Run ML model prediction on extracted features.
    Trains the model on training data, then predicts on video features.

    Args:
        features_df: DataFrame with all features

    Returns:
        Prediction results
    """
    print("\n" + "="*80)
    print("STAGE 3: MODEL PREDICTION")
    print("="*80)

    print("\n[1/4] Preparing features for model...")

    # Display input features
    print("\n   Input Features:")
    for col in features_df.columns:
        if col != 'video_name':
            print(f"     {col:20s}: {features_df[col].iloc[0]}")

    print("\n[2/4] Loading pre-trained ensemble model...")
    try:
        import pickle

        model_path = Path('trained_model.pkl')

        # Check if pre-trained model exists
        if model_path.exists():
            # Load pre-trained model (FAST!)
            with open(model_path, 'rb') as f:
                predictor = pickle.load(f)
            print("   [OK] Pre-trained model loaded from disk (10x faster!)")
        else:
            # Fallback: Train model from scratch (SLOW)
            print("   [!] Pre-trained model not found, training from scratch...")
            print("   [!] This is slow! Run 'python train_and_save_model.py' once to speed up future runs.")

            predictor = RunnerAdvancePredictor()
            print("   [OK] Model initialized")

            # Load training data
            train_df, test_df, position_df, glossary_df = predictor.load_data()
            print("   [OK] Training data loaded")

            # Preprocess data
            train_processed, test_processed = predictor.preprocess_data(train_df, test_df, position_df)
            print("   [OK] Data preprocessed")

            # Split features and target
            target_col = 'runner_advance'
            X_train = train_processed[predictor.feature_columns]
            y_train = train_processed[target_col]
            print(f"   [OK] Training set: {X_train.shape}, Target distribution: {y_train.value_counts().to_dict()}")

            # Train models
            predictor.train_models(X_train, y_train)
            print("   [OK] Models trained")

    except Exception as e:
        print(f"   [X] Error loading/training model: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n[3/4] Preparing video features for prediction...")

    # Prepare the feature DataFrame for prediction
    # Remove video_name column
    X_predict = features_df.drop('video_name', axis=1)

    # Encode categorical features using the same encoders from training
    for feature in ['fielder_pos', 'runner_base']:
        if feature in X_predict.columns and feature in predictor.label_encoders:
            le = predictor.label_encoders[feature]

            # Get the value
            value = X_predict[feature].iloc[0]

            # Check if the value is in the classes the encoder knows
            if pd.isna(value) or str(value) not in le.classes_:
                # Unseen value - use the most common class from training
                # For runner_base, default to 3 (most common in sacrifice flies)
                # For fielder_pos, default to 'CF' (most common)
                if feature == 'runner_base':
                    default_value = 3 if 3 in le.classes_ or '3' in le.classes_ else le.classes_[0]
                else:  # fielder_pos
                    default_value = 'CF' if 'CF' in le.classes_ else le.classes_[0]

                print(f"   [!] Warning: {feature}={value} not seen in training, using default={default_value}")
                X_predict[feature] = le.transform([str(default_value)])
            else:
                # Known value - encode it
                X_predict[feature] = le.transform([str(value)])

    # Ensure columns match training features
    if predictor.feature_columns is not None:
        # Reorder and select only the columns used in training
        X_predict = X_predict[predictor.feature_columns]

    # Handle NaN values - fill with reasonable defaults
    nan_defaults = {
        'hangtime': 4.0,  # Average hangtime for sac flies
        'player_distance_ft': 50.0,  # Reasonable default distance
        'exit_speed': 95.0,
        'launch_angle': 30.0,
        'hit_distance': 300.0,
        'exit_speed_squared': 9025.0,
        'launch_direction': 0.0
    }

    for col in X_predict.columns:
        if X_predict[col].isna().any():
            default_val = nan_defaults.get(col, 0)
            print(f"   [!] Warning: {col} has NaN, using default={default_val}")
            X_predict[col] = X_predict[col].fillna(default_val)

    print(f"   [OK] Feature vector prepared: {X_predict.shape}")

    print("\n[4/4] Making prediction with trained ensemble model...")

    try:
        # Get actual model prediction
        probability = predictor.predict(X_predict)[0]
        prediction = 1 if probability > 0.5 else 0

        print(f"   [OK] Prediction complete")
        print(f"   Model output: {probability:.6f} (raw probability)")

    except Exception as e:
        print(f"   [X] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        'prediction': prediction,
        'probability': probability
    }


def main():
    parser = argparse.ArgumentParser(
        description='Complete Baseball Pipeline: Video -> Tracking -> Features -> Prediction'
    )
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--metadata', required=True, help='Path to video_metadata.csv file')
    parser.add_argument('--output', default='pipeline_output', help='Output directory')

    args = parser.parse_args()

    # Get video name
    video_path = Path(args.video)
    video_name = video_path.stem
    video_filename = video_path.name

    print("="*80)
    print(f"COMPLETE PIPELINE: {video_name}")
    print("="*80)
    print(f"\nInputs:")
    print(f"  Video:    {args.video}")
    print(f"  Metadata: {args.metadata}")

    # Load metadata and find row for this video
    try:
        metadata_df = pd.read_csv(args.metadata, encoding='latin-1')
        video_row = metadata_df[metadata_df['video_filename'] == video_filename]

        if len(video_row) == 0:
            print(f"\n[X] ERROR: Video '{video_filename}' not found in metadata file")
            sys.exit(1)

        video_row = video_row.iloc[0]
        print(f"  Found metadata for: {video_filename}")

        # Create temporary statcast CSV for this video
        statcast_data = {
            'runner_base': video_row['runner_base'],
            'exit_speed': video_row['exit_velocity'],
            'launch_angle': video_row['launch_angle'],
            'hit_distance': video_row['hit_distance'],
            'hangtime': video_row['hangtime']
        }
        statcast_csv = f"temp_{video_name}_statcast.csv"
        pd.DataFrame([statcast_data]).to_csv(statcast_csv, index=False)
        print(f"  Statcast data extracted")

    except Exception as e:
        print(f"\n[X] ERROR loading metadata: {e}")
        sys.exit(1)

    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        # Stage 1: Run tracker on video
        tracker_csv = output_dir / f"{video_name}_tracker.csv"
        annotated_video = output_dir / f"{video_name}_annotated.mp4"
        run_tracker(args.video, str(tracker_csv), str(annotated_video))

        # Stage 2: Extract features
        features_df = extract_features(str(tracker_csv), statcast_csv, video_name)

        # Save features
        features_file = output_dir / f"{video_name}_features.csv"
        features_df.to_csv(features_file, index=False)
        print(f"\n   [OK] Features saved: {features_file}")

        # Stage 3: Run prediction
        results = run_model_prediction(features_df)

        if results:
            # Display results
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)

            print(f"\nVideo: {video_name}")
            print(f"Prediction: {'RUNNER ADVANCES (SAFE)' if results['prediction'] == 1 else 'RUNNER STAYS (OUT)'}")
            print(f"Probability: {results['probability']:.6f}")
            print(f"Confidence: {results['probability']:.1%}")

            # Save results
            results_file = output_dir / f"{video_name}_prediction.txt"
            with open(results_file, 'w') as f:
                f.write(f"Video: {video_name}\n")
                f.write(f"Prediction: {'ADVANCES' if results['prediction'] == 1 else 'STAYS'}\n")
                f.write(f"Probability: {results['probability']:.3f}\n")
                f.write(f"\nFeatures:\n")
                for col in features_df.columns:
                    if col != 'video_name':
                        f.write(f"  {col}: {features_df[col].iloc[0]}\n")

            print(f"\n[OK] Results saved: {results_file}")

            print("\n" + "="*80)
            print("PIPELINE COMPLETE [OK]")
            print("="*80)

        # Cleanup temporary statcast CSV
        if Path(statcast_csv).exists():
            Path(statcast_csv).unlink()

    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup temporary statcast CSV on error
        if 'statcast_csv' in locals() and Path(statcast_csv).exists():
            Path(statcast_csv).unlink()

        sys.exit(1)


if __name__ == '__main__':
    main()
