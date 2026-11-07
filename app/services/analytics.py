"""Industrial Edge Analytics: Directional Tripwire and Polygonal ROI Zone Monitoring.

Provides spatial analytics for tracked objects without external GIS dependencies.
"""

from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from app.models.schemas import TrackedObject


class TripwireDetector:
    """Directional virtual tripwire for counting entering and exiting objects."""

    def __init__(
        self,
        name: str,
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
    ) -> None:
        """Initialize tripwire with line endpoints (x1, y1) and (x2, y2)."""
        self.name = name
        self.p1 = np.array(line_start, dtype=np.float32)
        self.p2 = np.array(line_end, dtype=np.float32)

        self.in_count: int = 0
        self.out_count: int = 0
        self._prev_positions: Dict[int, np.ndarray] = {}
        self._crossed_ids: Set[int] = set()

    @staticmethod
    def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Counter-clockwise cross-product determinant."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _intersect(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        """Check if line segment AB intersects line segment CD."""
        ccw1 = self._ccw(a, b, c)
        ccw2 = self._ccw(a, b, d)
        ccw3 = self._ccw(c, d, a)
        ccw4 = self._ccw(c, d, b)
        return (
            ((ccw1 > 0 and ccw2 < 0) or (ccw1 < 0 and ccw2 > 0))
            and ((ccw3 > 0 and ccw4 < 0) or (ccw3 < 0 and ccw4 > 0))
        )

    def update(self, tracks: List[TrackedObject]) -> Dict[str, int]:
        """Update tracker positions and record directional crossing events."""
        current_ids = set()

        for track in tracks:
            tid = track.track_id
            current_ids.add(tid)

            # Use bottom-center point of bounding box (footprint anchor)
            cx = (track.bbox.x1 + track.bbox.x2) / 2.0
            cy = track.bbox.y2
            curr_pos = np.array([cx, cy], dtype=np.float32)

            if tid in self._prev_positions and tid not in self._crossed_ids:
                prev_pos = self._prev_positions[tid]
                if self._intersect(prev_pos, curr_pos, self.p1, self.p2):
                    # Determine crossing direction via determinant
                    det = self._ccw(self.p1, self.p2, curr_pos)
                    if det > 0:
                        self.in_count += 1
                    else:
                        self.out_count += 1
                    self._crossed_ids.add(tid)

            self._prev_positions[tid] = curr_pos

        # Clean up stale tracks
        stale_ids = set(self._prev_positions.keys()) - current_ids
        for sid in stale_ids:
            self._prev_positions.pop(sid, None)
            self._crossed_ids.discard(sid)

        return {
            "tripwire": self.name,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "total_count": self.in_count + self.out_count,
        }

    def draw(self, canvas: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        """Render virtual tripwire and counts on image canvas."""
        pt1 = (int(self.p1[0]), int(self.p1[1]))
        pt2 = (int(self.p2[0]), int(self.p2[1]))
        cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        label = f"{self.name} [In:{self.in_count} | Out:{self.out_count}]"
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2) - 8
        cv2.putText(canvas, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class PolygonZoneMonitor:
    """Polygonal Region of Interest (ROI) for dwell time and intrusion detection."""

    def __init__(self, name: str, polygon_points: List[Tuple[float, float]]) -> None:
        """Initialize ROI zone with polygon vertices."""
        self.name = name
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.current_occupancy: int = 0
        self.occupied_track_ids: List[int] = []

    def update(self, tracks: List[TrackedObject]) -> Dict[str, object]:
        """Check tracks against polygon ROI."""
        self.occupied_track_ids = []

        for track in tracks:
            # Bottom-center footprint anchor
            cx = (track.bbox.x1 + track.bbox.x2) / 2.0
            cy = track.bbox.y2
            # cv2.pointPolygonTest returns positive if inside, negative if outside, 0 if on edge
            dist = cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), False)
            if dist >= 0:
                self.occupied_track_ids.append(track.track_id)

        self.current_occupancy = len(self.occupied_track_ids)
        return {
            "zone": self.name,
            "occupancy": self.current_occupancy,
            "track_ids": self.occupied_track_ids,
        }

    def draw(self, canvas: np.ndarray, color: Tuple[int, int, int] = (255, 165, 0)) -> None:
        """Render polygon zone outline and occupancy label."""
        cv2.polylines(canvas, [self.polygon], isClosed=True, color=color, thickness=2)
        # Compute centroid for label
        moments = cv2.moments(self.polygon)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = int(self.polygon[0][0]), int(self.polygon[0][1])

        label = f"{self.name}: {self.current_occupancy} present"
        cv2.putText(canvas, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
