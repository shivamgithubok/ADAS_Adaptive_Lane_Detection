"""
test_occlusion_manager.py — Unit tests for the OcclusionManager module.

Run with:
    cd /home/elevatics/Projects/ADAS_Adaptive_Lane_Detection
    python -m pytest phase_03_lane/test_occlusion_manager.py -v
"""

import sys
from pathlib import Path

# Ensure imports resolve
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / 'phase_03_lane'))
sys.path.insert(0, str(base_dir / 'phase_02_geometry'))

import pytest
import numpy as np
from occlusion_manager import (
    OcclusionManager,
    OcclusionConfig,
    VehicleOcclusionState,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeMetricEstimate:
    def __init__(self, distance_m: float):
        self.distance_m = distance_m
        self.corrected_polygon = None
        self.vehicle_id = 0


class FakeGeometry:
    """Minimal stand-in for VehicleGeometry used by OcclusionManager."""
    def __init__(self, vid: int, distance: float):
        self.id = vid
        me = FakeMetricEstimate(distance)
        me.vehicle_id = vid
        self.metric_estimate = me
        self.footprint = True  # just truthy


def make_det(track_id: int, bbox, conf: float = 0.9):
    return {
        'track_id': track_id,
        'bbox': list(bbox),
        'conf': conf,
        'class_name': 'Car',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Static method tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIoU:
    def test_identical_boxes(self):
        box = (10, 10, 110, 110)
        assert OcclusionManager.compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        assert OcclusionManager.compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        # intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        assert OcclusionManager.compute_iou(a, b) == pytest.approx(2500 / 17500, rel=1e-3)

    def test_one_inside_other(self):
        outer = (0, 0, 200, 200)
        inner = (50, 50, 100, 100)
        inter = 50 * 50
        union = 200 * 200 + 50 * 50 - inter
        assert OcclusionManager.compute_iou(outer, inner) == pytest.approx(inter / union, rel=1e-3)


class TestBboxArea:
    def test_basic(self):
        assert OcclusionManager.compute_bbox_area((0, 0, 100, 50)) == 5000

    def test_zero_area(self):
        assert OcclusionManager.compute_bbox_area((10, 10, 10, 50)) == 0

    def test_negative_safe(self):
        assert OcclusionManager.compute_bbox_area((50, 50, 10, 10)) == 0


class TestOverlapRatio:
    def test_target_fully_covered(self):
        occluder = (0, 0, 200, 200)
        target = (50, 50, 100, 100)
        assert OcclusionManager.compute_overlap_ratio(occluder, target) == pytest.approx(1.0)

    def test_no_overlap(self):
        occluder = (0, 0, 50, 50)
        target = (100, 100, 200, 200)
        assert OcclusionManager.compute_overlap_ratio(occluder, target) == pytest.approx(0.0)

    def test_half_covered(self):
        occluder = (0, 0, 50, 100)
        target = (0, 0, 100, 100)
        # intersection = 50 * 100 = 5000, target area = 100 * 100 = 10000
        assert OcclusionManager.compute_overlap_ratio(occluder, target) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOcclusionManagerProcess:
    def setup_method(self):
        self.mgr = OcclusionManager()

    def test_empty_input(self):
        result = self.mgr.process([], [], 1280, 720)
        assert result == []

    def test_single_vehicle_fully_visible(self):
        geo = [FakeGeometry(1, 10.0)]
        det = [make_det(1, (100, 200, 300, 400))]
        result = self.mgr.process(geo, det, 1280, 720)

        assert len(result) == 1
        s = result[0]
        assert s.vehicle_id == 1
        assert s.visibility_score == pytest.approx(100.0, abs=0.1)
        assert s.render_mode == "FULL"
        assert s.planner_active is True
        assert s.tracking_state == "TRACKED"
        assert s.occluded_by == []

    def test_two_non_overlapping_vehicles(self):
        geo = [FakeGeometry(1, 5.0), FakeGeometry(2, 20.0)]
        det = [
            make_det(1, (100, 300, 300, 500)),  # near, left side
            make_det(2, (500, 100, 700, 200)),  # far, right side
        ]
        result = self.mgr.process(geo, det, 1280, 720)

        assert len(result) == 2
        # Both should be fully visible
        for s in result:
            assert s.visibility_score == pytest.approx(100.0, abs=0.1)
            assert s.render_mode == "FULL"
            assert s.planner_active is True

    def test_overlapping_vehicles_far_one_occluded(self):
        """Near vehicle partially covers far vehicle."""
        geo = [FakeGeometry(1, 5.0), FakeGeometry(2, 25.0)]
        det = [
            make_det(1, (100, 100, 300, 400)),  # near, large box
            make_det(2, (150, 150, 350, 350)),  # far, overlapping
        ]
        result = self.mgr.process(geo, det, 1280, 720)

        assert len(result) == 2
        near = next(s for s in result if s.vehicle_id == 1)
        far = next(s for s in result if s.vehicle_id == 2)

        # Near vehicle should be fully visible
        assert near.visibility_score == pytest.approx(100.0, abs=0.1)
        assert near.render_priority < far.render_priority

        # Far vehicle should be partially occluded
        assert far.visibility_score < 100.0
        assert far.occlusion_score > 0.0
        assert 1 in far.occluded_by

        # CRITICAL: far vehicle must NOT be removed
        assert far.planner_active is True

    def test_render_priority_depth_ordering(self):
        """Vehicles are sorted by distance: nearest gets render_priority 0."""
        geo = [FakeGeometry(3, 30.0), FakeGeometry(1, 5.0), FakeGeometry(2, 15.0)]
        det = [
            make_det(3, (100, 100, 200, 200)),
            make_det(1, (300, 300, 500, 500)),
            make_det(2, (400, 200, 600, 350)),
        ]
        result = self.mgr.process(geo, det, 1280, 720)

        priorities = {s.vehicle_id: s.render_priority for s in result}
        assert priorities[1] < priorities[2] < priorities[3]

    def test_planner_always_active(self):
        """No vehicle should ever have planner_active=False due to occlusion."""
        geo = [FakeGeometry(1, 5.0), FakeGeometry(2, 6.0)]
        det = [
            make_det(1, (100, 100, 400, 400)),
            make_det(2, (100, 100, 400, 400)),  # identical box → heavy occlusion
        ]
        result = self.mgr.process(geo, det, 1280, 720)

        for s in result:
            assert s.planner_active is True

    def test_occlusion_score_range(self):
        """Occlusion score must be in [0, 100]."""
        geo = [FakeGeometry(1, 5.0), FakeGeometry(2, 50.0)]
        det = [
            make_det(1, (0, 0, 500, 500)),
            make_det(2, (0, 0, 500, 500)),
        ]
        result = self.mgr.process(geo, det, 1280, 720)

        for s in result:
            assert 0.0 <= s.occlusion_score <= 100.0
            assert 0.0 <= s.visibility_score <= 100.0


class TestTemporalStability:
    def test_visibility_smoothing(self):
        """Visibility should be smoothed across frames, not jump abruptly."""
        mgr = OcclusionManager(OcclusionConfig(temporal_alpha=0.3))
        geo = [FakeGeometry(1, 10.0)]
        det = [make_det(1, (100, 100, 300, 300))]

        # First frame: fully visible
        r1 = mgr.process(geo, det, 1280, 720)
        assert r1[0].visibility_score == pytest.approx(100.0, abs=0.1)

        # Second frame: add an occluder that covers it significantly
        geo2 = [FakeGeometry(1, 20.0), FakeGeometry(2, 5.0)]
        det2 = [
            make_det(1, (100, 100, 300, 300)),
            make_det(2, (100, 100, 300, 300)),  # identical box → heavy occlusion
        ]
        r2 = mgr.process(geo2, det2, 1280, 720)
        far = next(s for s in r2 if s.vehicle_id == 1)

        # With EMA smoothing (alpha=0.3), visibility shouldn't drop to 0 immediately
        # It should be: 0.3 * raw + 0.7 * 100.0 → significantly above 0
        assert far.visibility_score > 20.0  # smoothed, not instant drop

    def test_stale_cleanup(self):
        """Temporal state for disappeared vehicles should be cleaned up."""
        mgr = OcclusionManager()
        geo = [FakeGeometry(1, 10.0)]
        det = [make_det(1, (100, 100, 300, 300))]

        mgr.process(geo, det, 1280, 720)
        assert 1 in mgr._seen_ids

        # Vehicle disappears
        mgr.process([], [], 1280, 720)
        assert 1 not in mgr._seen_ids


class TestRenderModeAssignment:
    def setup_method(self):
        self.mgr = OcclusionManager()

    def test_full_mode(self):
        assert self.mgr._assign_render_mode(100.0) == "FULL"
        assert self.mgr._assign_render_mode(80.0) == "FULL"

    def test_partial_mode(self):
        assert self.mgr._assign_render_mode(79.9) == "PARTIAL"
        assert self.mgr._assign_render_mode(50.0) == "PARTIAL"

    def test_outline_mode(self):
        assert self.mgr._assign_render_mode(49.9) == "OUTLINE"
        assert self.mgr._assign_render_mode(20.0) == "OUTLINE"

    def test_point_mode(self):
        assert self.mgr._assign_render_mode(19.9) == "POINT"
        assert self.mgr._assign_render_mode(0.1) == "POINT"

    def test_predicted_mode(self):
        assert self.mgr._assign_render_mode(0.0) == "PREDICTED"


class TestOutputSchema:
    """Verify every output field is present and correctly typed."""

    def test_all_fields_present(self):
        mgr = OcclusionManager()
        geo = [FakeGeometry(1, 10.0)]
        det = [make_det(1, (100, 200, 300, 400))]
        result = mgr.process(geo, det, 1280, 720)

        s = result[0]
        assert isinstance(s.vehicle_id, int)
        assert isinstance(s.distance, float)
        assert isinstance(s.bbox, tuple)
        assert len(s.bbox) == 4
        assert isinstance(s.visibility_score, float)
        assert isinstance(s.occlusion_score, float)
        assert isinstance(s.render_priority, int)
        assert isinstance(s.render_mode, str)
        assert isinstance(s.planner_active, bool)
        assert isinstance(s.tracking_state, str)
        assert s.geometry is not None
        assert isinstance(s.occluded_by, list)
        assert isinstance(s.frames_occluded, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
