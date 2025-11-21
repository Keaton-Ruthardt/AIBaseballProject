"""
Pixel-to-Feet Converter for Baseball Field

Converts pixel coordinates from video to real-world feet using baseball field dimensions.

Author: John Keaton Ruthardt
Date: November 2025
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseballFieldConverter:
    """
    Converts pixel coordinates to feet using baseball field geometry.

    Baseball field standard dimensions:
    - Bases: 90 feet apart
    - Home to center field wall: ~400 feet (varies by stadium)
    - Foul poles: ~330 feet
    - Typical outfield depth at catch: 300-400 feet
    """

    def __init__(
        self,
        video_width: int = 1280,
        video_height: int = 720,
        center_field_distance_ft: float = 400.0,
        foul_pole_distance_ft: float = 330.0
    ):
        """
        Initialize converter with video dimensions and field parameters.

        Args:
            video_width: Video width in pixels
            video_height: Video height in pixels
            center_field_distance_ft: Distance to center field wall (feet)
            foul_pole_distance_ft: Distance to foul poles (feet)
        """
        self.video_width = video_width
        self.video_height = video_height
        self.center_field_distance_ft = center_field_distance_ft
        self.foul_pole_distance_ft = foul_pole_distance_ft

        # Estimate home plate position (bottom center of frame)
        self.home_plate_px = (video_width // 2, video_height - 50)

        # Estimate outfield wall position (top 1/4 of frame)
        self.outfield_wall_y_px = video_height // 4

        # Calculate pixel-to-feet scaling factor
        # Vertical distance from home plate to outfield wall in pixels
        vertical_distance_px = self.home_plate_px[1] - self.outfield_wall_y_px

        # This represents ~400 feet in real world
        self.pixels_per_foot_vertical = vertical_distance_px / center_field_distance_ft

        # Horizontal scaling (assume similar, adjusted for perspective)
        # At outfield depth, 1 pixel ≈ same as vertical
        self.pixels_per_foot_horizontal = self.pixels_per_foot_vertical * 0.9

        logger.info(f"Converter initialized:")
        logger.info(f"  Video: {video_width}x{video_height}")
        logger.info(f"  Home plate (px): {self.home_plate_px}")
        logger.info(f"  Pixels per foot (vertical): {self.pixels_per_foot_vertical:.2f}")
        logger.info(f"  Pixels per foot (horizontal): {self.pixels_per_foot_horizontal:.2f}")

    def pixels_to_feet(
        self,
        x_px: float,
        y_px: float,
        reference_point: str = 'home_plate'
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to feet from reference point.

        Args:
            x_px: X coordinate in pixels
            y_px: Y coordinate in pixels
            reference_point: Reference point ('home_plate' or 'center_field')

        Returns:
            Tuple of (x_feet, y_feet) from reference point
        """
        if reference_point == 'home_plate':
            # Calculate offset from home plate
            dx_px = x_px - self.home_plate_px[0]
            dy_px = self.home_plate_px[1] - y_px  # Negative y in pixels = positive distance

            # Convert to feet
            x_feet = dx_px / self.pixels_per_foot_horizontal
            y_feet = dy_px / self.pixels_per_foot_vertical

            return (x_feet, y_feet)
        else:
            raise ValueError(f"Unknown reference point: {reference_point}")

    def distance_feet(self, x1_px: float, y1_px: float, x2_px: float, y2_px: float) -> float:
        """
        Calculate distance between two pixel points in feet.

        Args:
            x1_px, y1_px: First point in pixels
            x2_px, y2_px: Second point in pixels

        Returns:
            Distance in feet
        """
        # Convert both points to feet
        x1_ft, y1_ft = self.pixels_to_feet(x1_px, y1_px)
        x2_ft, y2_ft = self.pixels_to_feet(x2_px, y2_px)

        # Euclidean distance
        distance = np.sqrt((x2_ft - x1_ft)**2 + (y2_ft - y1_ft)**2)
        return distance

    def calculate_bearing(self, catch_x_px: float, catch_y_px: float) -> float:
        """
        Calculate bearing angle from home plate to catch position.

        Bearing is the horizontal angle from center field (0°) to the catch location.
        - 0° = straight to center field
        - -45° = left field line
        - +45° = right field line

        Args:
            catch_x_px: Catch X position in pixels
            catch_y_px: Catch Y position in pixels

        Returns:
            Bearing angle in degrees (-180 to +180)
        """
        # Convert to feet from home plate
        x_feet, y_feet = self.pixels_to_feet(catch_x_px, catch_y_px)

        # Calculate angle (arctangent)
        # 0° = straight away (center field)
        # Positive = right field, Negative = left field
        if y_feet == 0:
            bearing = 0.0
        else:
            bearing = np.arctan2(x_feet, y_feet)  # Note: x/y reversed for baseball bearing
            bearing = np.degrees(bearing)

        return bearing

    def calculate_launch_direction(self, catch_x_px: float, catch_y_px: float) -> float:
        """
        Calculate launch direction (similar to bearing but measured differently).

        Launch direction is typically measured as:
        - 0° = straight to center field
        - 90° = down right field line
        - -90° = down left field line

        Args:
            catch_x_px: Catch X position in pixels
            catch_y_px: Catch Y position in pixels

        Returns:
            Launch direction in degrees
        """
        # Use bearing calculation (same concept, different name)
        return self.calculate_bearing(catch_x_px, catch_y_px)

    def estimate_hit_distance_from_catch(
        self,
        catch_x_px: float,
        catch_y_px: float
    ) -> float:
        """
        Estimate hit distance from catch position (if not available from Statcast).

        Args:
            catch_x_px: Catch position X in pixels
            catch_y_px: Catch position Y in pixels

        Returns:
            Estimated hit distance in feet
        """
        # Distance from home plate to catch
        return self.distance_feet(
            self.home_plate_px[0], self.home_plate_px[1],
            catch_x_px, catch_y_px
        )

    def get_fielder_position_label(self, x_px: float, y_px: float) -> str:
        """
        Classify fielder position based on location.

        Args:
            x_px: Fielder X position in pixels
            y_px: Fielder Y position in pixels

        Returns:
            Position label: 'LF', 'CF', 'RF', or 'unknown'
        """
        # Convert to feet for better classification
        x_feet, y_feet = self.pixels_to_feet(x_px, y_px)

        # Must be in outfield depth (> 200 feet from home)
        distance = np.sqrt(x_feet**2 + y_feet**2)
        if distance < 200:
            return 'unknown'

        # Classify by horizontal position
        # Wider thresholds for more realistic field coverage
        # < -25° = LF
        # -25° to +25° = CF
        # > +25° = RF
        bearing = self.calculate_bearing(x_px, y_px)

        if bearing < -25:
            return 'LF'
        elif bearing > 25:
            return 'RF'
        else:
            return 'CF'

    def get_runner_base_from_position(self, x_px: float, y_px: float) -> int:
        """
        Estimate which base a runner is on/near.

        Args:
            x_px: Runner X position in pixels
            y_px: Runner Y position in pixels

        Returns:
            Base number: 1, 2, 3, or 0 (home/unknown)
        """
        x_feet, y_feet = self.pixels_to_feet(x_px, y_px)

        # Baseball diamond coordinates (90 feet between bases)
        # 1B: (90, 90) - right side
        # 2B: (0, 127) - straight away
        # 3B: (-90, 90) - left side

        bases = {
            1: (90, 90),
            2: (0, 127),
            3: (-90, 90)
        }

        # Find closest base
        min_distance = float('inf')
        closest_base = 0

        for base_num, (bx, by) in bases.items():
            distance = np.sqrt((x_feet - bx)**2 + (y_feet - by)**2)
            if distance < min_distance and distance < 50:  # Within 50 feet
                min_distance = distance
                closest_base = base_num

        return closest_base

    def convert_tracker_csv_to_feet(self, tracker_df) -> Dict:
        """
        Convert an entire tracker CSV to feet measurements.

        Args:
            tracker_df: DataFrame from simple_tracker.py output

        Returns:
            Dictionary with converted measurements
        """
        import pandas as pd

        results = {
            'frames': [],
            'players_feet': [],
            'catches': [],
            'distances_feet': []
        }

        for _, row in tracker_df.iterrows():
            frame_number = row['frame_number']

            # Convert each player's position
            player_positions = {}
            player_cols = [col for col in row.index if col.startswith('player_') and col.endswith('_x')]

            for col in player_cols:
                player_id = col.split('_')[1]
                x_col = f'player_{player_id}_x'
                y_col = f'player_{player_id}_y'

                if x_col in row and y_col in row:
                    x_px = row[x_col]
                    y_px = row[y_col]

                    if pd.notna(x_px) and pd.notna(y_px) and x_px != '':
                        x_ft, y_ft = self.pixels_to_feet(float(x_px), float(y_px))

                        player_positions[int(player_id)] = {
                            'x_feet': x_ft,
                            'y_feet': y_ft,
                            'distance_from_home': np.sqrt(x_ft**2 + y_ft**2),
                            'fielder_pos': self.get_fielder_position_label(float(x_px), float(y_px))
                        }

            results['frames'].append(frame_number)
            results['players_feet'].append(player_positions)

            # Check for catch event
            if 'catcher_player_id' in row and row['catcher_player_id'] != -1:
                catcher_id = int(row['catcher_player_id'])
                if catcher_id in player_positions:
                    # Get catch position in pixels
                    x_col = f'player_{catcher_id}_x'
                    y_col = f'player_{catcher_id}_y'
                    catch_x_px = float(row[x_col])
                    catch_y_px = float(row[y_col])

                    results['catches'].append({
                        'frame': frame_number,
                        'player_id': catcher_id,
                        'catch_x_feet': player_positions[catcher_id]['x_feet'],
                        'catch_y_feet': player_positions[catcher_id]['y_feet'],
                        'bearing': self.calculate_bearing(catch_x_px, catch_y_px),
                        'launch_direction': self.calculate_launch_direction(catch_x_px, catch_y_px),
                        'hit_distance_estimated': self.estimate_hit_distance_from_catch(catch_x_px, catch_y_px),
                        'fielder_pos': player_positions[catcher_id]['fielder_pos']
                    })

        return results


def test_converter():
    """Test the converter with sample data."""
    print("Testing Pixel-to-Feet Converter")
    print("=" * 60)

    # Initialize converter
    converter = BaseballFieldConverter(video_width=1280, video_height=720)

    # Test cases
    print("\nTest Cases:")
    print("-" * 60)

    # Test 1: Center field catch
    print("\n1. Center field catch (x=640, y=200):")
    x_ft, y_ft = converter.pixels_to_feet(640, 200)
    bearing = converter.calculate_bearing(640, 200)
    distance = np.sqrt(x_ft**2 + y_ft**2)
    fielder_pos = converter.get_fielder_position_label(640, 200)

    print(f"   Position (feet from home): ({x_ft:.1f}, {y_ft:.1f})")
    print(f"   Distance from home: {distance:.1f} feet")
    print(f"   Bearing: {bearing:.1f}°")
    print(f"   Fielder position: {fielder_pos}")

    # Test 2: Left field catch
    print("\n2. Left field catch (x=300, y=250):")
    x_ft, y_ft = converter.pixels_to_feet(300, 250)
    bearing = converter.calculate_bearing(300, 250)
    distance = np.sqrt(x_ft**2 + y_ft**2)
    fielder_pos = converter.get_fielder_position_label(300, 250)

    print(f"   Position (feet from home): ({x_ft:.1f}, {y_ft:.1f})")
    print(f"   Distance from home: {distance:.1f} feet")
    print(f"   Bearing: {bearing:.1f}°")
    print(f"   Fielder position: {fielder_pos}")

    # Test 3: Right field catch
    print("\n3. Right field catch (x=980, y=250):")
    x_ft, y_ft = converter.pixels_to_feet(980, 250)
    bearing = converter.calculate_bearing(980, 250)
    distance = np.sqrt(x_ft**2 + y_ft**2)
    fielder_pos = converter.get_fielder_position_label(980, 250)

    print(f"   Position (feet from home): ({x_ft:.1f}, {y_ft:.1f})")
    print(f"   Distance from home: {distance:.1f} feet")
    print(f"   Bearing: {bearing:.1f}°")
    print(f"   Fielder position: {fielder_pos}")

    print("\n" + "=" * 60)
    print("Converter test complete!")


if __name__ == "__main__":
    test_converter()
