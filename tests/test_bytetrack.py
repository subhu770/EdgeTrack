"""Unit and integration tests for ByteTrack multi-object tracking logic."""

import numpy as np
import pytest
from app.models.schemas import BoundingBox, Detection, TrackState
from app.services.bytetrack import ByteTracker, STrack, bbox_ious, linear_assignment


@pytest.fixture(autouse=True)
def reset_strack_counter() -> None:
    """Ensure clean track ID counter before each test."""
    STrack.reset_counter()


def test_track_initialization() -> None:
    """Test that high-confidence detections initialize active tracks."""
    tracker = ByteTracker(
        track_high_thresh=0.5,
        new_track_thresh=0.6,
        min_consecutive_frames=1,
    )

    det = Detection(
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=200.0),
        confidence=0.85,
        class_id=0,
        class_name="person",
    )

    active_tracks = tracker.update([det])

    assert len(active_tracks) == 1
    assert active_tracks[0].track_id == 1
    assert active_tracks[0].state == TrackState.TRACKED
    assert active_tracks[0].confidence == 0.85
    assert active_tracks[0].class_name == "person"


def test_two_stage_low_score_association() -> None:
    """Test ByteTrack 2nd stage association: matching low-confidence detection.

    Frame 1: Object detected with high confidence (0.90) -> Track initialized.
    Frame 2: Object is partially occluded, confidence drops to 0.30 (< 0.5 threshold).
             Standard trackers discard <0.5 detections, losing track.
             ByteTrack 2nd stage matches low-score detection to confirmed track!
    """
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.8,
        match_thresh_second=0.5,
        min_consecutive_frames=1,
    )

    # Frame 1: High confidence detection
    d1 = Detection(
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=200.0),
        confidence=0.90,
        class_id=0,
    )
    tracks_f1 = tracker.update([d1])
    assert len(tracks_f1) == 1
    track_id = tracks_f1[0].track_id

    # Frame 2: Slight movement, but low confidence detection (e.g. occlusion)
    d2 = Detection(
        bbox=BoundingBox(x1=104.0, y1=100.0, x2=154.0, y2=200.0),
        confidence=0.30,  # Below track_high_thresh (0.5), above track_low_thresh (0.1)
        class_id=0,
    )
    tracks_f2 = tracker.update([d2])

    assert len(tracks_f2) == 1
    # ID must be strictly preserved via 2nd stage association!
    assert tracks_f2[0].track_id == track_id
    assert pytest.approx(tracks_f2[0].bbox.x1, abs=2.0) == 104.0


def test_lost_and_removed_lifecycle() -> None:
    """Test track lifecycle from Tracked -> Lost -> Removed after max_lost_frames."""
    max_lost = 3
    tracker = ByteTracker(
        track_high_thresh=0.5,
        max_lost_frames=max_lost,
        min_consecutive_frames=1,
    )

    # Frame 1: Detection activates track
    d1 = Detection(
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=90.0, y2=90.0),
        confidence=0.85,
        class_id=0,
    )
    tracks_f1 = tracker.update([d1])
    assert len(tracks_f1) == 1
    assert len(tracker.tracked_stracks) == 1

    # Frames 2, 3, 4: Zero detections (object disappears)
    tracker.update([])
    assert len(tracker.tracked_stracks) == 0
    assert len(tracker.lost_stracks) == 1
    assert tracker.lost_stracks[0].state == TrackState.LOST

    tracker.update([])
    assert len(tracker.lost_stracks) == 1

    tracker.update([])
    assert len(tracker.lost_stracks) == 1

    # Frame 5: Exceeds max_lost_frames (3) -> Track removed permanently
    tracker.update([])
    assert len(tracker.lost_stracks) == 0
    assert len(tracker.removed_stracks) >= 1
    assert tracker.removed_stracks[0].state == TrackState.REMOVED


def test_track_reidentification_after_being_lost() -> None:
    """Test that a lost track is re-identified and reactivated when detection returns."""
    tracker = ByteTracker(
        track_high_thresh=0.5,
        max_lost_frames=10,
        min_consecutive_frames=1,
    )

    # Frame 1: Object appears
    d1 = Detection(
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=150.0),
        confidence=0.90,
        class_id=0,
    )
    t1 = tracker.update([d1])[0]
    orig_id = t1.track_id

    # Frame 2: Object missing for 1 frame -> becomes LOST
    tracker.update([])
    assert len(tracker.lost_stracks) == 1
    assert tracker.lost_stracks[0].track_id == orig_id

    # Frame 3: Object reappears at predicted position
    d3 = Detection(
        bbox=BoundingBox(x1=102.0, y1=100.0, x2=152.0, y2=150.0),
        confidence=0.88,
        class_id=0,
    )
    t3 = tracker.update([d3])

    assert len(t3) == 1
    # Track re-identified with original ID!
    assert t3[0].track_id == orig_id
    assert t3[0].state == TrackState.TRACKED


def test_occlusion_id_persistence_crossing_trajectories() -> None:
    """Test two objects crossing paths and verify persistent IDs without ID switches."""
    tracker = ByteTracker(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        min_consecutive_frames=1,
    )

    # Object 1: starts at x=100, moves right (+5 px/frame)
    # Object 2: starts at x=300, moves left (-5 px/frame)
    # They cross around frame 20 (x=200)
    for frame_idx in range(1, 41):
        x1 = 100.0 + 5.0 * frame_idx
        x2 = 300.0 - 5.0 * frame_idx

        det1 = Detection(
            bbox=BoundingBox(x1=x1, y1=150.0, x2=x1 + 40.0, y2=230.0),
            confidence=0.90,
            class_id=0,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=x2, y1=160.0, x2=x2 + 40.0, y2=240.0),
            confidence=0.90,
            class_id=0,
        )

        tracks = tracker.update([det1, det2])
        assert len(tracks) == 2

        # Check track IDs
        ids = {t.track_id for t in tracks}
        assert ids == {1, 2}, f"Frame {frame_idx}: IDs mutated to {ids}"

        # Verify velocities maintain opposite signs
        trk_dict = {t.track_id: t for t in tracks}
        if frame_idx > 5:
            assert trk_dict[1].velocity.vx > 0  # moving right
            assert trk_dict[2].velocity.vx < 0  # moving left

