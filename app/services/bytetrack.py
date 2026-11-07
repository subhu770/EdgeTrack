"""ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

Pure NumPy/SciPy implementation with Kalman Filter state estimation and
two-stage Hungarian data association.

Reference:
    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
    ECCV 2022.
"""

from typing import List, Optional, Tuple
import numpy as np
import scipy.optimize

from app.models.schemas import BoundingBox, Detection, TrackState, TrackedObject, Velocity
from app.services.kalman import KalmanFilterTracker


class STrack:
    """Single Object Tracklet maintaining trajectory state, Kalman estimation, and lifecycle."""

    _count: int = 0

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        class_id: int,
        class_name: Optional[str] = None,
    ) -> None:
        """Initialize tracklet from detection bounding box."""
        # [top_left_x, top_left_y, width, height]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = KalmanFilterTracker()
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated: bool = False

        self.score: float = float(score)
        self.class_id: int = int(class_id)
        self.class_name: Optional[str] = class_name
        self.tracklet_len: int = 0

        self.track_id: int = 0
        self.state: TrackState = TrackState.NEW
        self.frame_id: int = 0
        self.start_frame: int = 0
        self.time_since_update: int = 0

    @classmethod
    def reset_counter(cls) -> None:
        """Reset global track ID counter (for simulation or new stream)."""
        cls._count = 0

    @classmethod
    def next_id(cls) -> int:
        """Generate monotonically increasing unique track ID."""
        cls._count += 1
        return cls._count

    @property
    def end_frame(self) -> int:
        """Return the frame where the track was last observed."""
        return self.frame_id

    @property
    def tlwh(self) -> np.ndarray:
        """Get bounding box in [top_left_x, top_left_y, width, height] format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # w = aspect_ratio * height
        ret[:2] -= ret[2:] / 2.0  # tl_x = cx - w/2, tl_y = cy - h/2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box in [x1, y1, x2, y2] format."""
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def velocity(self) -> Optional[Velocity]:
        """Estimated horizontal and vertical velocity in px/frame."""
        if self.mean is not None and len(self.mean) >= 6:
            return Velocity(vx=round(float(self.mean[4]), 2), vy=round(float(self.mean[5]), 2))
        return None

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert [tl_x, tl_y, w, h] to [center_x, center_y, aspect_ratio, height]."""
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= max(ret[3], 1e-6)
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert current tracklet bounding box to xyah."""
        return self.tlwh_to_xyah(self.tlwh)

    def activate(self, kalman_filter: KalmanFilterTracker, frame_id: int) -> None:
        """Activate tracklet with unique ID and initialize Kalman state."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def re_activate(
        self, new_track: "STrack", frame_id: int, new_id: bool = False
    ) -> None:
        """Re-activate a lost track with a new matching detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = new_track.score
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """Update track with associated detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score

    def predict(self) -> None:
        """Predict bounding box at next time step using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean_state[7] = 0.0  # damp height velocity for unconfirmed or lost tracks
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    def mark_lost(self) -> None:
        """Transition tracklet state to LOST."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Transition tracklet state to REMOVED."""
        self.state = TrackState.REMOVED

    def to_tracked_object(self) -> TrackedObject:
        """Convert STrack to Pydantic TrackedObject schema."""
        x1, y1, x2, y2 = self.tlbr
        return TrackedObject(
            track_id=self.track_id,
            bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            confidence=round(self.score, 4),
            class_id=self.class_id,
            class_name=self.class_name,
            state=self.state,
            age=self.frame_id - self.start_frame + 1,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
        )


def bbox_ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Calculate IoU distance matrix between two collections of boxes.

    Args:
        atlbrs: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        btlbrs: Array of shape (M, 4) in [x1, y1, x2, y2] format.

    Returns:
        Matrix of shape (N, M) containing IoU values in [0.0, 1.0].
    """
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    x1 = np.maximum(atlbrs[:, None, 0], btlbrs[None, :, 0])
    y1 = np.maximum(atlbrs[:, None, 1], btlbrs[None, :, 1])
    x2 = np.minimum(atlbrs[:, None, 2], btlbrs[None, :, 2])
    y2 = np.minimum(atlbrs[:, None, 3], btlbrs[None, :, 3])

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter_area = w * h

    area_a = np.maximum(0.0, atlbrs[:, 2] - atlbrs[:, 0]) * np.maximum(
        0.0, atlbrs[:, 3] - atlbrs[:, 1]
    )
    area_b = np.maximum(0.0, btlbrs[:, 2] - btlbrs[:, 0]) * np.maximum(
        0.0, btlbrs[:, 3] - btlbrs[:, 1]
    )
    union_area = area_a[:, None] + area_b[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-7)

    return (inter_area / union_area).astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Solve linear sum assignment problem with cost threshold gating.

    Args:
        cost_matrix: 2D cost matrix (e.g. 1.0 - IoU) of shape (N, M).
        thresh: Maximum allowed cost to consider a match valid.

    Returns:
        matches: Array of shape (K, 2) with valid (row, col) matches.
        unmatched_a: Indices of unmatched rows.
        unmatched_b: Indices of unmatched columns.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = []
    unmatched_a: List[int] = []
    unmatched_b: List[int] = []

    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows and r not in unmatched_a:
            unmatched_a.append(r)

    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols and c not in unmatched_b:
            unmatched_b.append(c)

    matches_arr = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches_arr, unmatched_a, unmatched_b


class ByteTracker:
    """ByteTrack Multi-Object Tracker.

    Manages active, lost, and removed tracks using two-stage Hungarian association.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        match_thresh_second: float = 0.5,
        match_thresh_unconfirmed: float = 0.7,
        max_lost_frames: int = 30,
        min_consecutive_frames: int = 2,
    ) -> None:
        """Initialize ByteTrack parameters."""
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_second = match_thresh_second
        self.match_thresh_unconfirmed = match_thresh_unconfirmed
        self.max_lost_frames = max_lost_frames
        self.min_consecutive_frames = min_consecutive_frames

        self.kalman_filter = KalmanFilterTracker()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id: int = 0

    def reset(self) -> None:
        """Reset tracker state and clear all tracks."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_counter()

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Process detections for the current frame and return active tracks.

        Args:
            detections: List of Detection objects with bbox and confidence.

        Returns:
            List of TrackedObject representing currently active, confirmed tracks.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        # Convert detections to STrack instances
        dets_high: List[STrack] = []
        dets_low: List[STrack] = []

        for det in detections:
            bbox = det.bbox
            tlwh = np.array([bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1], dtype=np.float32)
            track = STrack(tlwh, det.confidence, det.class_id, det.class_name)

            if det.confidence >= self.track_high_thresh:
                dets_high.append(track)
            elif det.confidence >= self.track_low_thresh:
                dets_low.append(track)

        # Partition existing tracks into confirmed vs unconfirmed
        unconfirmed_stracks: List[STrack] = []
        tracked_stracks: List[STrack] = []

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                tracked_stracks.append(track)

        # Track pool for 1st association: confirmed tracked + lost tracks
        strack_pool: List[STrack] = tracked_stracks + self.lost_stracks

        # Predict current locations for all existing tracks
        for track in strack_pool:
            track.predict()
            track.time_since_update += 1

        for track in unconfirmed_stracks:
            track.predict()
            track.time_since_update += 1

        # -------------------------------------------------------------
        # Stage 1: Association between high-score detections and track pool
        # -------------------------------------------------------------
        cost_matrix_1 = np.empty((len(strack_pool), len(dets_high)), dtype=np.float32)
        if len(strack_pool) > 0 and len(dets_high) > 0:
            pool_boxes = np.array([t.tlbr for t in strack_pool], dtype=np.float32)
            det_boxes = np.array([d.tlbr for d in dets_high], dtype=np.float32)
            ious = bbox_ious(pool_boxes, det_boxes)
            cost_matrix_1 = 1.0 - ious

        matches_1, unmatched_tracks_1, unmatched_dets_1 = linear_assignment(
            cost_matrix_1, thresh=self.match_thresh
        )

        for track_idx, det_idx in matches_1:
            track = strack_pool[track_idx]
            det = dets_high[det_idx]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # -------------------------------------------------------------
        # Stage 2: Association between low-score detections and remaining tracks
        # -------------------------------------------------------------
        # Only tracks that were previously TRACKED and unmatched in Stage 1
        remain_tracked_indices = [
            i for i in unmatched_tracks_1 if strack_pool[i].state == TrackState.TRACKED
        ]
        remain_tracked_stracks = [strack_pool[i] for i in remain_tracked_indices]

        cost_matrix_2 = np.empty(
            (len(remain_tracked_stracks), len(dets_low)), dtype=np.float32
        )
        if len(remain_tracked_stracks) > 0 and len(dets_low) > 0:
            remain_boxes = np.array([t.tlbr for t in remain_tracked_stracks], dtype=np.float32)
            low_det_boxes = np.array([d.tlbr for d in dets_low], dtype=np.float32)
            ious_2 = bbox_ious(remain_boxes, low_det_boxes)
            cost_matrix_2 = 1.0 - ious_2

        matches_2, unmatched_tracks_2, _ = linear_assignment(
            cost_matrix_2, thresh=self.match_thresh_second
        )

        for track_idx, det_idx in matches_2:
            track = remain_tracked_stracks[track_idx]
            det = dets_low[det_idx]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Tracks unmatched after Stage 2 are marked LOST
        for track_idx in unmatched_tracks_2:
            track = remain_tracked_stracks[track_idx]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with lost tracks that remained unmatched in Stage 1
        for i in unmatched_tracks_1:
            track = strack_pool[i]
            if track.state == TrackState.LOST and track not in lost_stracks:
                lost_stracks.append(track)

        # -------------------------------------------------------------
        # Stage 3: Deal with unconfirmed tracks & unmatched high detections
        # -------------------------------------------------------------
        unmatched_dets_high = [dets_high[i] for i in unmatched_dets_1]

        cost_matrix_3 = np.empty(
            (len(unconfirmed_stracks), len(unmatched_dets_high)), dtype=np.float32
        )
        if len(unconfirmed_stracks) > 0 and len(unmatched_dets_high) > 0:
            unconfirmed_boxes = np.array([t.tlbr for t in unconfirmed_stracks], dtype=np.float32)
            det_boxes_3 = np.array([d.tlbr for d in unmatched_dets_high], dtype=np.float32)
            ious_3 = bbox_ious(unconfirmed_boxes, det_boxes_3)
            cost_matrix_3 = 1.0 - ious_3

        matches_3, unmatched_unconfirmed, unmatched_dets_3 = linear_assignment(
            cost_matrix_3, thresh=self.match_thresh_unconfirmed
        )

        for track_idx, det_idx in matches_3:
            track = unconfirmed_stracks[track_idx]
            det = unmatched_dets_high[det_idx]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # Unmatched unconfirmed tracks are removed
        for track_idx in unmatched_unconfirmed:
            track = unconfirmed_stracks[track_idx]
            track.mark_removed()
            removed_stracks.append(track)

        # Initialize new tracks from remaining unmatched high-score detections
        for det_idx in unmatched_dets_3:
            det = unmatched_dets_high[det_idx]
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                if self.min_consecutive_frames <= 1:
                    det.is_activated = True
                    activated_stracks.append(det)
                else:
                    det.is_activated = False
                    unconfirmed_stracks.append(det)

        # -------------------------------------------------------------
        # Stage 4: Update lost and removed tracks
        # -------------------------------------------------------------
        # Consolidate candidate lost tracks from this frame
        candidate_lost = [
            t for t in (self.lost_stracks + lost_stracks) if t.state == TrackState.LOST
        ]

        # Check for expiration exceeding max_lost_frames
        final_lost: List[STrack] = []
        for track in candidate_lost:
            if self.frame_id - track.end_frame > self.max_lost_frames:
                track.mark_removed()
                removed_stracks.append(track)
            else:
                final_lost.append(track)

        # Consolidate tracker state
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.TRACKED
        ]
        self.tracked_stracks = list(
            {t.track_id: t for t in (self.tracked_stracks + activated_stracks + refind_stracks)}.values()
        )

        # Update lost tracks list (excluding reactivated and removed)
        self.lost_stracks = [
            t for t in final_lost
            if t not in activated_stracks and t not in refind_stracks and t not in removed_stracks
        ]
        self.lost_stracks = list({t.track_id: t for t in self.lost_stracks}.values())

        # Include unconfirmed tracks in tracked_stracks for continuity
        for track in unconfirmed_stracks:
            if track.state != TrackState.REMOVED and track not in self.tracked_stracks:
                self.tracked_stracks.append(track)

        self.removed_stracks.extend(removed_stracks)

        # Filter and return only active, confirmed tracks
        output_stracks = [
            t for t in self.tracked_stracks if t.is_activated and t.state == TrackState.TRACKED
        ]
        return [t.to_tracked_object() for t in output_stracks]
