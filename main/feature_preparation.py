"""
Feature Preparation Module

Combines tracker CSV data with Statcast CSV data to prepare features
for the ensemble ML model.

Handles:
1. Tracker CSVs (from simple_tracker.py) - pixel coordinates, player positions
2. Statcast CSVs (user-provided) - hit metrics (exit_velocity, launch_angle, etc.)
3. Pixel-to-feet conversion
4. Bearing calculation
5. Feature engineering for model input

Author: John Keaton Ruthardt
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from pixel_to_feet_converter import BaseballFieldConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePreparation:
    """
    Prepares features from tracker + Statcast data for ML model.
    """

    def __init__(self, video_width: int = 1280, video_height: int = 720):
        """
        Initialize feature preparation.

        Args:
            video_width: Video width in pixels
            video_height: Video height in pixels
        """
        self.converter = BaseballFieldConverter(video_width, video_height)
        logger.info("Feature preparation initialized")

    def load_tracker_csv(self, tracker_csv_path: str) -> pd.DataFrame:
        """
        Load tracker CSV from simple_tracker.py output.

        Args:
            tracker_csv_path: Path to tracker CSV

        Returns:
            DataFrame with tracker data
        """
        df = pd.read_csv(tracker_csv_path)
        logger.info(f"Loaded tracker CSV: {tracker_csv_path}")
        logger.info(f"  Frames: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        return df

    def load_statcast_csv(self, statcast_csv_path: str) -> pd.DataFrame:
        """
        Load Statcast CSV with hit metrics.

        Expected columns:
        - runner_base
        - exit_velocity (or exit_speed)
        - launch_angle
        - hit_distance
        - hangtime

        Args:
            statcast_csv_path: Path to Statcast CSV

        Returns:
            DataFrame with Statcast data
        """
        df = pd.read_csv(statcast_csv_path)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle alternative names
        if 'exit_velocity' in df.columns and 'exit_speed' not in df.columns:
            df['exit_speed'] = df['exit_velocity']

        logger.info(f"Loaded Statcast CSV: {statcast_csv_path}")
        logger.info(f"  Rows: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")

        return df

    def extract_catch_frame_features(self, tracker_df: pd.DataFrame) -> Dict:
        """
        Extract features from the catch frame in tracker data.

        Args:
            tracker_df: Tracker DataFrame

        Returns:
            Dictionary with catch frame features
        """
        # Find catch frame (where catcher_player_id >= 0)
        catch_mask = pd.to_numeric(tracker_df['catcher_player_id'], errors='coerce').fillna(-1) >= 0

        if not catch_mask.any():
            logger.warning("No catch event found in tracker data")
            return {}

        # Get first catch frame
        catch_row = tracker_df[catch_mask].iloc[0]
        catcher_id = int(catch_row['catcher_player_id'])

        # Get catcher position
        catch_x_col = f'player_{catcher_id}_x'
        catch_y_col = f'player_{catcher_id}_y'

        if catch_x_col not in catch_row or catch_y_col not in catch_row:
            logger.warning(f"Catcher position not found for player {catcher_id}")
            return {}

        catch_x_px = float(catch_row[catch_x_col])
        catch_y_px = float(catch_row[catch_y_col])

        # Convert to feet and calculate metrics
        converted = self.converter.convert_tracker_csv_to_feet(pd.DataFrame([catch_row]))

        if converted['catches']:
            catch_data = converted['catches'][0]

            features = {
                'catch_frame': int(catch_row['frame_number']),
                'catcher_player_id': catcher_id,
                'catch_x_feet': catch_data['catch_x_feet'],
                'catch_y_feet': catch_data['catch_y_feet'],
                'bearing': catch_data['bearing'],
                'launch_direction': catch_data['launch_direction'],
                'hit_distance_estimated': catch_data['hit_distance_estimated'],
                'fielder_pos': catch_data['fielder_pos']
            }

            # Calculate outfield spread if multiple players
            player_positions = converted['players_feet'][0]
            if len(player_positions) >= 2:
                # Get all distances between players
                distances = []
                player_ids = list(player_positions.keys())
                for i in range(len(player_ids)):
                    for j in range(i+1, len(player_ids)):
                        p1 = player_positions[player_ids[i]]
                        p2 = player_positions[player_ids[j]]
                        dx = p1['x_feet'] - p2['x_feet']
                        dy = p1['y_feet'] - p2['y_feet']
                        dist = np.sqrt(dx**2 + dy**2)
                        distances.append(dist)

                if distances:
                    features['outfield_spread_event_4'] = max(distances)  # Max spread
                else:
                    features['outfield_spread_event_4'] = None
            else:
                features['outfield_spread_event_4'] = None

            logger.info(f"Extracted catch frame features:")
            logger.info(f"  Frame: {features['catch_frame']}")
            logger.info(f"  Fielder: {features['fielder_pos']}")
            logger.info(f"  Bearing: {features['bearing']:.1f}°")
            logger.info(f"  Estimated distance: {features['hit_distance_estimated']:.1f} ft")

            return features
        else:
            return {}

    def prepare_model_features(
        self,
        tracker_csv_path: str,
        statcast_csv_path: str,
        video_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare complete feature set for ML model.

        Combines tracker + Statcast data into model-ready format.

        Args:
            tracker_csv_path: Path to tracker CSV
            statcast_csv_path: Path to Statcast CSV
            video_name: Optional video name for identification

        Returns:
            DataFrame with all required features
        """
        logger.info(f"Preparing features for: {video_name or 'video'}")

        # Load data
        tracker_df = self.load_tracker_csv(tracker_csv_path)
        statcast_df = self.load_statcast_csv(statcast_csv_path)

        # Extract catch frame features from tracker
        catch_features = self.extract_catch_frame_features(tracker_df)

        # Build feature row
        features = {}

        # Add video identifier
        if video_name:
            features['video_name'] = video_name
        else:
            features['video_name'] = Path(tracker_csv_path).stem

        # --- FROM STATCAST CSV ---
        # These are the 6 guaranteed features from Statcast
        if len(statcast_df) > 0:
            row = statcast_df.iloc[0]  # Take first row

            features['hit_distance'] = row.get('hit_distance', None)
            features['exit_speed'] = row.get('exit_speed', row.get('exit_velocity', None))
            features['launch_angle'] = row.get('launch_angle', None)
            features['hangtime'] = row.get('hangtime', None)
            features['runner_base'] = row.get('runner_base', 3)  # Default to 3rd base

            # Calculate exit_speed_squared
            if features['exit_speed'] is not None:
                features['exit_speed_squared'] = features['exit_speed'] ** 2
            else:
                features['exit_speed_squared'] = None

        # --- FROM TRACKER CSV (with pixel conversion) ---
        if catch_features:
            # Bearing - calculated from catch position
            features['bearing'] = catch_features['bearing']

            # Launch direction - same as bearing for our purposes
            features['launch_direction'] = catch_features['launch_direction']

            # Fielder position - who caught the ball (LF/CF/RF)
            features['fielder_pos'] = catch_features['fielder_pos']

            # Outfield spread (OPTIONAL - model works without it)
            features['outfield_spread_event_4'] = catch_features.get('outfield_spread_event_4', None)

            # If hit_distance not in Statcast, use estimated from tracker
            if features.get('hit_distance') is None:
                features['hit_distance'] = catch_features['hit_distance_estimated']
                logger.info(f"Using estimated hit_distance: {features['hit_distance']:.1f} ft")
        else:
            # No catch detected - use defaults
            logger.warning("No catch detected, using default values")
            features['bearing'] = 0.0
            features['launch_direction'] = 0.0
            features['fielder_pos'] = 'CF'
            features['outfield_spread_event_4'] = None

        # Create DataFrame
        features_df = pd.DataFrame([features])

        logger.info(f"Features prepared:")
        logger.info(f"  Total features: {len(features)}")
        for key, value in features.items():
            if pd.notna(value):
                logger.info(f"  {key}: {value}")

        return features_df

    def prepare_batch_features(
        self,
        tracker_csv_dir: str,
        statcast_csv_dir: str
    ) -> pd.DataFrame:
        """
        Prepare features for multiple videos in batch.

        Matches tracker CSVs with corresponding Statcast CSVs by filename.

        Args:
            tracker_csv_dir: Directory with tracker CSVs
            statcast_csv_dir: Directory with Statcast CSVs

        Returns:
            DataFrame with features for all videos
        """
        tracker_dir = Path(tracker_csv_dir)
        statcast_dir = Path(statcast_csv_dir)

        all_features = []

        # Find all tracker CSVs
        tracker_csvs = list(tracker_dir.glob('*.csv'))
        logger.info(f"Found {len(tracker_csvs)} tracker CSVs")

        for tracker_csv in tracker_csvs:
            video_name = tracker_csv.stem

            # Find matching Statcast CSV
            statcast_csv = statcast_dir / f"{video_name}.csv"

            if not statcast_csv.exists():
                # Try alternative naming
                statcast_csv = statcast_dir / f"{video_name}_statcast.csv"

            if not statcast_csv.exists():
                logger.warning(f"No Statcast CSV found for {video_name}, skipping")
                continue

            # Prepare features for this video
            try:
                features_df = self.prepare_model_features(
                    str(tracker_csv),
                    str(statcast_csv),
                    video_name
                )
                all_features.append(features_df)
                logger.info(f"✓ Processed: {video_name}")

            except Exception as e:
                logger.error(f"✗ Failed to process {video_name}: {e}")
                continue

        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"\nBatch processing complete:")
            logger.info(f"  Videos processed: {len(all_features)}")
            logger.info(f"  Features prepared: {len(combined_df.columns)}")
            return combined_df
        else:
            logger.error("No features prepared")
            return pd.DataFrame()


def test_feature_preparation():
    """Test feature preparation with sample data."""
    print("Testing Feature Preparation")
    print("=" * 60)

    # Initialize
    prep = FeaturePreparation()

    # Check if sample files exist
    tracker_csv = "simple_output.csv"
    statcast_csv = "sample_statcast.csv"  # You'll need to create this

    if not Path(tracker_csv).exists():
        print(f"ERROR: {tracker_csv} not found")
        print("Please run simple_tracker.py first to generate tracker CSV")
        return

    if not Path(statcast_csv).exists():
        print(f"WARNING: {statcast_csv} not found")
        print("Creating sample Statcast CSV for testing...")

        # Create sample Statcast data
        sample_statcast = pd.DataFrame({
            'runner_base': [3],
            'exit_speed': [92.5],
            'launch_angle': [28.0],
            'hit_distance': [350.0],
            'hangtime': [4.8]
        })
        sample_statcast.to_csv(statcast_csv, index=False)
        print(f"Created: {statcast_csv}")

    # Test feature preparation
    features_df = prep.prepare_model_features(tracker_csv, statcast_csv, "test_video")

    print("\nPrepared Features:")
    print("-" * 60)
    print(features_df.T)

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_feature_preparation()
