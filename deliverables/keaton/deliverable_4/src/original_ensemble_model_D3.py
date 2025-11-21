"""
Baseball Runner Advancement Prediction Model

This module implements a machine learning pipeline to predict the probability
of a runner successfully advancing on sacrifice plays in baseball. The model
uses game situation features, player performance metrics, and ball trajectory
data to make predictions.

Author: John Keaton Ruthardt
Purpose: Interview Assessment - Baseball Analytics

"""

import pandas as pd
import numpy as np
import os, re, glob # New Import 
from pathlib import Path # New Import 
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class RunnerAdvancePredictor:
    """
    This class handles data preprocessing, feature engineering, model training,
    and prediction generation for baseball sacrifice play scenarios. Built with OOP in
    mind. Allows for reuseablity with the purpose of creating a professional real deployable
    model. Multiple models allows for different situations, seasons, and opponents to compare. 
    """

    def __init__(self):
        """Constructor - Initialize the predictor with default configuration. """
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = None
        self.is_trained = False

    def load_data(self): # New updates
        """
        Load and return all dataset components.
        The new changes allow the model to run smoothly even when the outfielder distance data is missing, 
        automatically adjusting between 9 and 10 features.

        Returns:
            tuple: (train_df, test_df, position_df, glossary_df)
        """
        print("Loading datasets...")

        # Load main datasets
        train_df = pd.read_csv('train_data.csv')
        test_df = pd.read_csv('test_data.csv')
        position_df = None # We need to check if outfield_position exist
        glossary_df = pd.read_csv('glossary.csv')

        #Check if outfield_position exist
        if Path('outfield_position.csv').exists():
            try:
                position_df = pd.read_csv('outfield_position.csv')
            except Exception:
                position_df = None

        # If the position file does not exist or is empty, try to build it from tracker CSVs
        if position_df is None or position_df.empty or not set(['play_id','event_code']).issubset(set(map(str.lower, position_df.columns.str.lower()))):
            # Look for typical tracker CSV files
            candidates = []

            # Include the sample file if present
            if Path('simple_output.csv').exists():
                candidates.append('simple_output.csv')

            # Also look for more CSVs inside deliverables/tracker or the root directory
            candidates.extend(glob.glob('deliverables/tracker/*.csv'))
            candidates.extend(glob.glob('*.csv'))

            # Filter candidates that appear to come from the tracker
            # (contain columns like player_* or catcher_player_id)
            tracker_like = []
            for c in candidates:
                try:
                    _df = pd.read_csv(c, nrows=1)
                    has_player = any(re.match(r'^player_\d+_(x|y)$', col) for col in _df.columns)
                    has_catch = 'catcher_player_id' in _df.columns
                    if has_player or has_catch:
                        tracker_like.append(c)
                except Exception:
                    continue

            if tracker_like:
                # Build the position dataframe from any detected tracker CSVs
                position_df = self.position_df_from_tracker_csvs(tracker_like)
            else:
                # Create an empty DataFrame with minimal required columns
                position_df = pd.DataFrame(columns=['play_id','event_code','pos_x','pos_y'])

        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Position data shape: {position_df.shape}")

        return train_df, test_df, position_df, glossary_df


    def create_position_features(self, df, position_df):
        """
        Engineer features from outfielder position tracking data. Converting 
        GPS coordinates into defensive positioning for features. 

        Args:
            df: Main dataframe with play data
            position_df: Outfielder position tracking data

        Returns:
            pd.DataFrame: Dataframe with added position features
        """
        print("Engineering position-based features...")

        # Aggregate position data by play_id and event
        position_agg = position_df.groupby(['play_id', 'event_code']).agg({
            'pos_x': ['mean', 'std', 'min', 'max'],
            'pos_y': ['mean', 'std', 'min', 'max']
        }).reset_index()

        # Flatten column names
        position_agg.columns = ['play_id', 'event_code'] + [
            f'pos_{col[0]}_{col[1]}' for col in position_agg.columns[2:]
        ]

        # Calculate outfield spread (distance between fielders)
        position_spread = position_df.groupby(['play_id', 'event_code']).apply(
            lambda x: np.sqrt((x['pos_x'].max() - x['pos_x'].min())**2 +
                             (x['pos_y'].max() - x['pos_y'].min())**2)
        ).reset_index(name='outfield_spread')

        # Merge position features for different events.
        # Iterates through two critical moments in each sacrifice play.
        # The difference between these two moments can reveal how fielders moved during 
        # the play, which will impact advancement probability.
        for event_code in [3, 4]:  # Ball hit and Ball caught events
            event_data = position_agg[position_agg['event_code'] == event_code].copy()
            event_data = event_data.drop('event_code', axis=1)
            event_data.columns = ['play_id'] + [f'{col}_event_{event_code}'
                                               for col in event_data.columns[1:]]
            df = df.merge(event_data, on='play_id', how='left')

            # Add spread data
            spread_data = position_spread[position_spread['event_code'] == event_code].copy()
            spread_data = spread_data.drop('event_code', axis=1)
            spread_data.columns = ['play_id', f'outfield_spread_event_{event_code}']
            df = df.merge(spread_data, on='play_id', how='left')

        return df

    def engineer_features(self, df):
        """
        Create additional features from existing data.

        Args:
            df: Input dataframe

        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("Engineering additional features...")

        # Score differential (how close the game is)
        df['score_diff'] = df['home_score'] - df['away_score']
        df['score_diff_abs'] = abs(df['score_diff'])

        # Game pressure indicators (removed late_inning and high_leverage due to low importance)

        # Ball trajectory features
        df['exit_speed_squared'] = df['exit_speed'] ** 2
        df['launch_angle_optimal'] = ((df['launch_angle'] >= 15) &
                                     (df['launch_angle'] <= 35)).astype(int)

        # Distance and hang time interaction
        df['distance_hangtime_ratio'] = df['hit_distance'] / (df['hangtime'] + 0.001)

        # Player performance features
        df['fielder_throw_advantage'] = df['fielder_max_throwspeed'] - df['fielder_max_throwspeed'].median()
        df['runner_speed_advantage'] = df['runner_max_sprintspeed'] - df['runner_max_sprintspeed'].median()

        # Count and situation features (removed specific count indicators due to low importance)

        # Enhanced features for improved performance
        # Multiple baserunner context
        df['total_baserunners'] = (~df['pre_runner_1b'].isna()).astype(int) + \
                                 (~df['pre_runner_2b'].isna()).astype(int) + \
                                 (~df['pre_runner_3b'].isna()).astype(int)
        df['runner_on_first'] = (~df['pre_runner_1b'].isna()).astype(int)
        df['multiple_runners'] = (df['total_baserunners'] > 1).astype(int)

        # Platoon advantage
        df['platoon_advantage'] = (df['batter_side'] != df['pitcher_side']).astype(int)

        # Enhanced count features
        df['behind_in_count'] = (df['strikes'] > df['balls']).astype(int)
        df['pressure_count'] = ((df['strikes'] >= 2) | (df['balls'] >= 3)).astype(int)

        # Game flow features (removed due to low importance)

        # Spin-based features
        df['spin_efficiency'] = df['launch_spinrate'] / (df['exit_speed'] + 0.001)

        # Missing value indicators (capture information about missingness patterns)
        # These variables are most prone to equipment failure, correlated with play type,
        # and strategically relevant
        # Example: "When exit_speed_missing = 1 AND hangtime_missing = 1 --> 85% chance its a short pop up --> 
        # Short pop-ups have ...% advancement prevention rate" This provides more information then random imputation.
        missing_features = ['exit_speed', 'launch_angle', 'hit_distance', 'hangtime',
                           'fielder_max_throwspeed', 'launch_spinrate', 'bearing']

        for feature in missing_features:
            if feature in df.columns:
                df[f'{feature}_missing'] = df[feature].isnull().astype(int)

        return df

    def preprocess_data(self, train_df, test_df, position_df):
        """
        Complete data preprocessing pipeline.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            position_df: Position tracking dataframe

        Returns:
            tuple: (processed_train_df, processed_test_df)
        """
        print("Starting data preprocessing pipeline...")

        # Add position features to both datasets
        train_df = self.create_position_features(train_df, position_df)
        test_df = self.create_position_features(test_df, position_df)

        # Engineer additional features
        train_df = self.engineer_features(train_df)
        test_df = self.engineer_features(test_df)

        # Select numerical and categorical features (optimized feature set)
        numerical_features = [
            'hit_distance',          # | 20.3% | Statcast Overlay | Read from screen or measure visually
            #'launch_direction',     # REMOVED - not consistent from video analysis
            'exit_speed_squared',    # |  2.9% | Cal ted       | exit_speed * exit_speed
            'exit_speed',            # |  2.5% | Statcast Overlay | Read from screen (MPH)
            'hangtime',              # |  2.1% | Statcast/Video   | Read from screen or time from hit to catch
            'launch_angle'           # |  1.9% | Statcast Overlay | Read from screen  (degrees)
           #'outfield_spread_event_4'  Test if it still works with one feature less
        ]
        
        categorical_features = [
            'fielder_pos',           # |  2.3% | Computer Vision  | Detect which fielder catches (LF/CF/RF) |
            'runner_base'            # | 24.2% | Computer Vision  | Track runner position at pitch (1B/2B/3B) |
        ]

        # These section was commented out to avoid automatically adding all position-related features
        # Add position features to numerical list 
        # position_features = [col for col in train_df.columns if 'pos_' in col or 'outfield_spread' in col]
        # numerical_features.extend(position_features) 
        # Handle missing values in numerical features


        num_imputer = SimpleImputer(strategy='median')
        train_df[numerical_features] = num_imputer.fit_transform(train_df[numerical_features])
        test_df[numerical_features] = num_imputer.transform(test_df[numerical_features])

        # Encode categorical variables
        for feature in categorical_features:
            if feature in train_df.columns:
                le = LabelEncoder()
                # Fit on combined data to handle unseen categories
                combined_values = pd.concat([train_df[feature], test_df[feature]]).fillna('missing')
                le.fit(combined_values)

                train_df[feature] = le.transform(train_df[feature].fillna('missing'))
                test_df[feature] = le.transform(test_df[feature].fillna('missing'))

                self.label_encoders[feature] = le

        # Store feature columns for later use
        self.feature_columns = numerical_features + categorical_features

        print(f"Final feature count: {len(self.feature_columns)}")
        print(f"Features: {self.feature_columns}") #Show the features
        return train_df, test_df

    def train_models(self, X_train, y_train):
        """
        Train multiple models and evaluate their performance. These different alogirthms catch 
        different patterns, leading to their combined prediction being more reliable than any single
        model

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training machine learning models...")

        # Define model configurations
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }

        # Split data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)
        self.scalers['standard'] = scaler

        # Train and evaluate each model
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in model_configs.items():
            print(f"\nTraining {name}...")

            if name == 'logistic_regression':
                # Use scaled features for logistic regression
                model.fit(X_train_scaled, y_train_split)
                val_probs = model.predict_proba(X_val_scaled)[:, 1]

                # Cross-validation with scaled features
                X_train_full_scaled = scaler.fit_transform(X_train)
                cv_scores = cross_val_score(model, X_train_full_scaled, y_train,
                                          cv=cv_folds, scoring='neg_log_loss')
            else:
                # Use original features for tree-based models
                model.fit(X_train_split, y_train_split)
                val_probs = model.predict_proba(X_val_split)[:, 1]

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train,
                                          cv=cv_folds, scoring='neg_log_loss')

            # Calculate validation metrics
            val_log_loss = log_loss(y_val_split, val_probs)
            val_auc = roc_auc_score(y_val_split, val_probs)

            print(f"{name} - Validation Log Loss: {val_log_loss:.4f}")
            print(f"{name} - Validation AUC: {val_auc:.4f}")
            print(f"{name} - CV Log Loss: {-cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

            self.models[name] = model

        self.is_trained = True
        print("\nModel training completed!")

    def predict(self, X_test):
        """
        Generate ensemble predictions from trained models.

        Args:
            X_test: Test features

        Returns:
            np.array: Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        print("Generating predictions...")
        predictions = {}

        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'logistic_regression':
                X_test_scaled = self.scalers['standard'].transform(X_test)
                predictions[name] = model.predict_proba(X_test_scaled)[:, 1]
            else:
                predictions[name] = model.predict_proba(X_test)[:, 1]

        # Create ensemble prediction (weighted average)
        # Tree-based models get higher weight due to better validation performance
        weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.4,
            'logistic_regression': 0.2
        }

        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred

        return ensemble_pred

    def get_feature_importance(self):
        """
        Extract and display feature importance from tree-based models.

        Returns:
            pd.DataFrame: Feature importance summary
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before extracting feature importance")

        importance_data = []

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    importance_data.append({
                        'model': name,
                        'feature': self.feature_columns[i],
                        'importance': importance
                    })

        importance_df = pd.DataFrame(importance_data)

        # Calculate average importance across models
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)

        print("\n 10 Most Important Features:")
        print("=" * 50)
        for i, (feature, importance) in enumerate(avg_importance.head(15).items(), 1):
            print(f"{i:2d}. {feature:<30} {importance:.4f}")

        return importance_df

    def position_df_from_tracker_csvs(self, csv_paths): # New Section
        """
        This new function converts the tracker’s wide-format CSV into a clean, long-format DataFrame.
        """
        rows = []
        player_re = re.compile(r"^player_(\d+)_(x|y)$") # Regex to capture player IDs and coordinate types

        for p in csv_paths:
            p = Path(p)
            if not p.exists():
                continue # Skip missing files
            play_id = p.stem  # Use the file name
            try:
                df = pd.read_csv(p)
            except Exception:
                continue # Skip unreadable CSVs

            # Skip files that don't contain frame information
            if 'frame_number' not in df.columns:
                continue

            # Identify which frame to use: ideally the frame where the catch occurs
            catch_frame = None
            if 'catcher_player_id' in df.columns:
                mask = pd.to_numeric(df['catcher_player_id'], errors='coerce').fillna(-1) >= 0
                if mask.any():
                    catch_frame = int(df.loc[mask, 'frame_number'].iloc[0])

            # If no catch was detected, fall back to the frame with the most players visible
            if catch_frame is None:
                if 'num_outfielders_detected' in df.columns:
                    idx = df['num_outfielders_detected'].fillna(0).astype(float).idxmax()
                    catch_frame = int(df.loc[idx, 'frame_number'])
                else:
                    # As a last resort, use the very first frame
                    catch_frame = int(df['frame_number'].iloc[0])

            # Get the data from the selected frame
            row = df[df['frame_number'] == catch_frame]
            if row.empty:
                continue
            row = row.iloc[0].to_dict()
            event_code = 4 # Use event_code=4 to represent the “catch moment”

            # Collect all player coordinate pairs from that frame
            player_cols = {}
            for col in list(row.keys()):
                m = player_re.match(col)
                if not m:
                    continue
                pid, axis = m.groups()
                d = player_cols.setdefault(int(pid), {})
                d[axis] = row[col]

            # Convert each player’s (x, y) position into one row of the position_df
            for pid, axes in player_cols.items():
                x = axes.get('x', '')
                y = axes.get('y', '')
                if x == '' or y == '' or pd.isna(x) or pd.isna(y):
                    continue # Skip missing or invalid positions
                try:
                    px = float(x); py = float(y)
                except Exception:
                    continue # Skip non-numeric data

                rows.append({
                'play_id': play_id,         # From the filename
                'event_code': event_code,   # 4 = catch event
                'pos_x': px,                # X-coordinate of the player
                'pos_y': py                 # Y-coordinate of the player
                })

        # If no valid rows were created, return an empty DataFrame with correct columns
        if not rows:
            return pd.DataFrame(columns=['play_id','event_code','pos_x','pos_y'])

        # Return the final DataFrame ready to merge with training/test data
        return pd.DataFrame(rows)


def main():
    """
    Main execution function for the runner advancement prediction pipeline.
    """
    print("Baseball Runner Advancement Prediction Pipeline")
    print("=" * 60)

    # Initialize predictor
    predictor = RunnerAdvancePredictor()

    # Load data
    train_df, test_df, position_df, glossary_df = predictor.load_data()

    # Preprocess data
    train_processed, test_processed = predictor.preprocess_data(train_df, test_df, position_df)

    # Prepare training data
    X_train = train_processed[predictor.feature_columns]
    y_train = train_processed['runner_advance']
    X_test = test_processed[predictor.feature_columns]

    print(f"\nTraining set class distribution:")
    print(f"No advancement (0): {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
    print(f"Successful advancement (1): {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")

    # Train models
    predictor.train_models(X_train, y_train)

    # Display feature importance
    predictor.get_feature_importance()

    # Generate predictions
    test_predictions = predictor.predict(X_test)

    # Create submission dataframe
    submission_df = test_df[['play_id']].copy()
    submission_df['runner_advance'] = test_predictions

    # Save predictions
    submission_df.to_csv('runner_advance_predictions.csv', index=False)

    print(f"\nPredictions saved to 'runner_advance_predictions.csv'")
    print(f"Prediction statistics:")
    print(f"Mean probability: {test_predictions.mean():.3f}")
    print(f"Min probability: {test_predictions.min():.3f}")
    print(f"Max probability: {test_predictions.max():.3f}")
    print(f"Std deviation: {test_predictions.std():.3f}")

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
    
    