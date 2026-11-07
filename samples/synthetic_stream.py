"""Synthetic Moving Objects Stream & Multi-Object Tracking Benchmark.

Generates synthetic video canvas with crossing objects undergoing occlusion,
motion blur, and confidence drops to benchmark ByteTrack ID persistence
without requiring external videos or GPU acceleration.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.models.schemas import BoundingBox, Detection
from app.services.bytetrack import ByteTracker, STrack


class SyntheticObject:
    """Simulated moving object with trajectory and occlusion parameters."""

    def __init__(
        self,
        name: str,
        start_pos: Tuple[float, float],
        velocity: Tuple[float, float],
        size: Tuple[float, float],
        color: Tuple[int, int, int],
        class_id: int = 0,
        class_name: str = "person",
        conf_drop_interval: Tuple[int, int] = (-1, -1),
    ) -> None:
        self.name = name
        self.x, self.y = start_pos
        self.vx, self.vy = velocity
        self.w, self.h = size
        self.color = color
        self.class_id = class_id
        self.class_name = class_name
        self.conf_drop_interval = conf_drop_interval

    def update(self, frame_idx: int) -> Tuple[BoundingBox, float]:
        """Update position and return current BoundingBox and confidence score."""
        self.x += self.vx
        self.y += self.vy

        confidence = 0.92
        start_drop, end_drop = self.conf_drop_interval
        if start_drop <= frame_idx <= end_drop:
            # Simulate heavy occlusion or blur: confidence drops to 0.25 (below 0.5 high threshold)
            confidence = 0.25

        bbox = BoundingBox(
            x1=self.x,
            y1=self.y,
            x2=self.x + self.w,
            y2=self.y + self.h,
        )
        return bbox, confidence


def run_synthetic_benchmark(
    num_frames: int = 100,
    width: int = 640,
    height: int = 480,
    save_video: bool = False,
    video_output: str = "samples/synthetic_tracking_demo.avi",
) -> None:
    """Execute synthetic tracking simulation and benchmark ID persistence."""
    print("=" * 75)
    print("EdgeTrack: High-Performance Synthetic Multi-Object Tracking Benchmark")
    print("=" * 75)

    # Initialize Tracker
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.8,
        match_thresh_second=0.5,
        max_lost_frames=30,
        min_consecutive_frames=1,
    )
    STrack.reset_counter()

    # Set deterministic seed for repeatable benchmarking
    np.random.seed(42)

    # Define targets:
    # Target-A: Moves left to right
    # Target-B: Moves right to left (crosses Target-A with severe spatial overlap around frame 45-55)
    # Target-C: Moves diagonally with severe confidence drop between frames 30-38
    objects = [
        SyntheticObject(
            name="Target-A (Left->Right)",
            start_pos=(50.0, 200.0),
            velocity=(5.0, 0.0),
            size=(50.0, 100.0),
            color=(255, 100, 50),  # Blueish
            class_id=0,
            class_name="person",
        ),
        SyntheticObject(
            name="Target-B (Right->Left)",
            start_pos=(540.0, 215.0),
            velocity=(-5.0, 0.0),
            size=(50.0, 100.0),
            color=(50, 100, 255),  # Reddish
            class_id=0,
            class_name="person",
        ),
        SyntheticObject(
            name="Target-C (Diagonal)",
            start_pos=(80.0, 50.0),
            velocity=(4.5, 3.5),
            size=(60.0, 60.0),
            color=(50, 220, 50),   # Green
            class_id=2,
            class_name="car",
            conf_drop_interval=(30, 38),
        ),
    ]

    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(video_output, fourcc, 25.0, (width, height))

    target_history: Dict[str, List[Tuple[int, int]]] = {obj.name: [] for obj in objects}
    latencies: List[float] = []

    print(f"\n[1/3] Simulating {num_frames} frames with real-time ByteTrack association...")

    for frame_idx in range(1, num_frames + 1):
        # Create dark canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Draw grid
        for gx in range(0, width, 80):
            cv2.line(canvas, (gx, 0), (gx, height), (45, 45, 45), 1)
        for gy in range(0, height, 80):
            cv2.line(canvas, (0, gy), (width, gy), (45, 45, 45), 1)

        detections: List[Detection] = []

        # Render ground truth objects & generate synthetic detections
        for obj in objects:
            bbox, conf = obj.update(frame_idx)

            # Skip rendering/detection if off canvas
            if bbox.x1 > width or bbox.x2 < 0 or bbox.y1 > height or bbox.y2 < 0:
                continue

            # Draw ground truth rectangle on canvas
            cv2.rectangle(
                canvas,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                obj.color,
                2,
            )

            # Add synthetic detection (with slight sensor noise)
            noisy_bbox = BoundingBox(
                x1=bbox.x1 + np.random.uniform(-1.0, 1.0),
                y1=bbox.y1 + np.random.uniform(-1.0, 1.0),
                x2=bbox.x2 + np.random.uniform(-1.0, 1.0),
                y2=bbox.y2 + np.random.uniform(-1.0, 1.0),
            )
            detections.append(
                Detection(
                    bbox=noisy_bbox,
                    confidence=conf,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                )
            )

        # Execute ByteTrack update
        t0 = time.perf_counter()
        active_tracks = tracker.update(detections)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Match active tracks to ground truth objects via Hungarian matching for accurate ID persistence audit
        if objects and active_tracks:
            gt_boxes = np.array(
                [[obj.x, obj.y, obj.x + obj.w, obj.y + obj.h] for obj in objects],
                dtype=np.float32,
            )
            trk_boxes = np.array(
                [[t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2] for t in active_tracks],
                dtype=np.float32,
            )
            from app.services.bytetrack import bbox_ious, linear_assignment

            cost_matrix = 1.0 - bbox_ious(gt_boxes, trk_boxes)
            matches, _, _ = linear_assignment(cost_matrix, thresh=0.6)
            for gt_idx, trk_idx in matches:
                target_history[objects[gt_idx].name].append(
                    (frame_idx, active_tracks[trk_idx].track_id)
                )

        # Annotate tracked bounding boxes
        for trk in active_tracks:
            b = trk.bbox
            cv2.rectangle(
                canvas,
                (int(b.x1), int(b.y1)),
                (int(b.x2), int(b.y2)),
                (0, 255, 255),
                2,
            )
            label = f"ID:{trk.track_id} {trk.class_name} [{trk.confidence:.2f}]"
            cv2.putText(
                canvas,
                label,
                (int(b.x1), max(18, int(b.y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Overlay frame info
        info_text = f"Frame: {frame_idx:03d} | Active Tracks: {len(active_tracks)} | Latency: {latencies[-1]:.3f}ms"
        cv2.putText(
            canvas, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        if video_writer is not None:
            video_writer.write(canvas)

    if video_writer is not None:
        video_writer.release()
        print(f"Annotated benchmark video saved to '{video_output}'.")

    # Evaluation & Verification
    print("\n[2/3] Verification of ID Persistence Across Occlusion:")
    print("-" * 75)
    id_switches = 0
    for name, history in target_history.items():
        if not history:
            print(f"  {name}: No tracks recorded.")
            continue
        assigned_ids = [tid for _, tid in history]
        unique_ids = set(assigned_ids)
        primary_id = max(unique_ids, key=assigned_ids.count)
        switches = len(unique_ids) - 1
        id_switches += switches
        frames_tracked = len(history)

        status = "PASSED (Zero ID Switches)" if switches == 0 else f"FAILED ({switches} ID Switches)"
        print(f"  {name:<26} | Tracked Frames: {frames_tracked:02d}/{num_frames} | Track ID: {primary_id} | {status}")

    print("-" * 75)
    print(f"Total Track ID Switches: {id_switches}")

    # Latency summary
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n[3/3] EdgeTrack Pure NumPy/SciPy Performance Breakdown:")
    print("-" * 75)
    print(f"  Mean Tracking Latency : {avg_latency:.4f} ms per frame")
    print(f"  P95 Tracking Latency  : {p95_latency:.4f} ms per frame")
    print(f"  Effective Throughput  : {fps:.1f} FPS (CPU Single-Thread)")
    print("=" * 75)

    if id_switches == 0:
        print("SUCCESS: 100% ID persistence maintained across crossing trajectories & occlusion!")
    else:
        print("WARNING: Track ID switches detected during trajectory crossing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeTrack Synthetic Stream Benchmark")
    parser.add_argument("--frames", type=int, default=100, help="Number of simulation frames")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video demo")
    args = parser.parse_args()

    run_synthetic_benchmark(num_frames=args.frames, save_video=args.save_video)
