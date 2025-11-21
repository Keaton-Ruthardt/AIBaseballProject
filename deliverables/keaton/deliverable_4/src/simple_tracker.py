"""
Simplified Outfielder Tracker

Outputs simple CSV with:
- Frame number
- Outfielder positions (pixel coordinates)
- Who caught the ball
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional
from collections import OrderedDict
from scipy.spatial import distance as dist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCentroidTracker:
    """Simple centroid tracker for maintaining player IDs."""

    def __init__(self, max_disappeared: int = 30, max_distance: int = 150):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox_history = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: np.ndarray, bbox: List[int]):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bbox_history[self.next_object_id] = bbox
        self.next_object_id += 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bbox_history[object_id]

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        input_centroids = np.array([d['center'] for d in detections])
        input_bboxes = [d['bbox'] for d in detections]

        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            D = dist.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bbox_history[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])

        tracked_objects = {}
        for object_id in self.objects.keys():
            tracked_objects[object_id] = {
                'center': self.objects[object_id].tolist(),
                'bbox': self.bbox_history[object_id]
            }
        return tracked_objects


class SimpleOutfielderTracker:
    """
    Simplified tracker that only detects real outfielders.
    Stricter filtering to avoid false positives.
    """

    def __init__(
        self,
        model_size: str = 'n',
        confidence_threshold: float = 0.25,
        device: str = 'cpu'
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device

        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading {model_name}")
        self.model = YOLO(model_name)

        self.tracker = SimpleCentroidTracker(max_disappeared=30, max_distance=150)
        self.prev_frame = None

    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """Detect all people in frame."""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=0.3,
            classes=[0],
            verbose=False,
            device=self.device,
            imgsz=1280
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx, cy],
                    'confidence': conf,
                    'area': area
                })

        return detections

    def filter_to_real_outfielders(
        self,
        detections: List[Dict],
        frame_shape: tuple
    ) -> List[Dict]:
        """
        STRICT filtering to only get real outfielders in the field.

        Rules:
        - Must be in MIDDLE vertical region (not top where fans/scoreboards are)
        - Must be reasonable size (not huge closeups, not tiny distant objects)
        - Must be in field area (centered horizontally)
        """
        height, width = frame_shape

        outfielders = []
        for det in detections:
            cx, cy = det['center']
            area = det['area']

            # STRICT RULES to avoid false positives:

            # 1. Must be in MIDDLE of frame vertically (30% - 75%)
            #    This excludes fans/scoreboards at top (y < 30%)
            if not (height * 0.30 < cy < height * 0.75):
                continue

            # 2. Size constraints
            if area < 3000 or area > 35000:
                continue

            # 3. Must be reasonably centered horizontally (10% - 90%)
            if not (width * 0.10 < cx < width * 0.90):
                continue

            outfielders.append(det)

        return outfielders

    def detect_ball(self, frame: np.ndarray) -> Optional[List[int]]:
        """Detect ball position."""
        results = self.model.predict(
            frame,
            conf=0.15,
            iou=0.5,
            classes=[32],  # Sports ball
            verbose=False,
            device=self.device,
            imgsz=1280
        )

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            return [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        return None

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """Process single frame."""
        height, width = frame.shape[:2]

        # Detect all people
        all_detections = self.detect_players(frame)

        # Filter to real outfielders only
        outfielder_detections = self.filter_to_real_outfielders(
            all_detections, (height, width)
        )

        # Track outfielders
        tracked_players = self.tracker.update(outfielder_detections)

        # Detect ball
        ball_position = self.detect_ball(frame)

        # Determine catcher
        catcher_id = None
        catcher_distance = None

        if ball_position and len(tracked_players) > 0:
            min_dist = float('inf')
            closest_id = None

            for player_id, player_data in tracked_players.items():
                px, py = player_data['center']
                bx, by = ball_position

                distance = np.sqrt((px - bx)**2 + (py - by)**2)
                if distance < min_dist:
                    min_dist = distance
                    closest_id = player_id

            # Only count as catch if within 120 pixels
            if min_dist < 120:
                catcher_id = closest_id
                catcher_distance = min_dist

        self.prev_frame = frame.copy()

        return {
            'frame_number': frame_number,
            'tracked_players': tracked_players,
            'ball_position': ball_position,
            'catcher_id': catcher_id,
            'catcher_distance': catcher_distance
        }

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[Dict]:
        """Process video and return results."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        results = []
        frame_idx = 0

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if end_frame and frame_idx >= end_frame:
                    break

                result = self.process_frame(frame, frame_idx)
                results.append(result)

                # Visualize
                if writer:
                    vis = frame.copy()

                    # Draw tracked players
                    for player_id, player_data in result['tracked_players'].items():
                        x1, y1, x2, y2 = player_data['bbox']
                        cx, cy = player_data['center']

                        color = (0, 255, 0)  # Green
                        if result['catcher_id'] == player_id:
                            color = (0, 255, 255)  # Yellow for catcher

                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, f"Player {player_id}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.circle(vis, (cx, cy), 5, color, -1)

                    # Draw ball
                    if result['ball_position']:
                        bx, by = result['ball_position']
                        cv2.circle(vis, (bx, by), 10, (0, 255, 255), 3)

                    # Draw catch event
                    if result['catcher_id'] is not None:
                        cv2.putText(vis, f"CATCH by Player {result['catcher_id']}",
                                   (10, vis.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                    writer.write(vis)

                if frame_idx % 30 == 0:
                    logger.info(f"Frame {frame_idx}: {len(result['tracked_players'])} players tracked")

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
                logger.info(f"Saved: {output_path}")

        logger.info(f"Processed {len(results)} frames")
        return results

    def export_simple_csv(self, results: List[Dict], output_csv: str):
        """
        Export SIMPLE CSV with just:
        - Frame number
        - Outfielder positions (x, y for each player)
        - Who caught the ball
        """
        import pandas as pd

        rows = []
        for r in results:
            row = {
                'frame_number': r['frame_number'],
                'num_outfielders_detected': len(r['tracked_players']),
            }

            # Add each tracked player's position
            player_ids = sorted(r['tracked_players'].keys())
            for player_id in player_ids:
                player_data = r['tracked_players'][player_id]
                row[f'player_{player_id}_x'] = player_data['center'][0]
                row[f'player_{player_id}_y'] = player_data['center'][1]

            # Catch info - who caught the ball
            row['catcher_player_id'] = r['catcher_id'] if r['catcher_id'] is not None else -1

            rows.append(row)

        df = pd.DataFrame(rows)

        # Fill NaN with empty for player positions that don't exist
        df = df.fillna('')

        df.to_csv(output_csv, index=False)
        logger.info(f"Exported to: {output_csv}")

        # Summary
        players_tracked = set()
        for r in results:
            players_tracked.update(r['tracked_players'].keys())

        logger.info(f"Total unique players tracked: {len(players_tracked)}")
        logger.info(f"Player IDs: {sorted(players_tracked)}")

        return df


if __name__ == "__main__":
    print("Simple Outfielder Tracker loaded")
