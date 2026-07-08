"""
occlusion_manager.py — Production-grade Occlusion Management for ADAS Pipeline

Inserted between Metric Projection and ODZ Filter.
Never removes valid tracked vehicles. Estimates visibility, occlusion,
and rendering priority. Separates planner data from visualization decisions.

Pipeline position:
    YOLO → Tracker → Metric Projection → **OcclusionManager** → ODZ → Viz → Planning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class OcclusionConfig:
    """All tunable thresholds for occlusion management."""

    # Pairwise analysis triggers
    iou_threshold: float = 0.15          # IoU above which occlusion analysis fires
    overlap_threshold: float = 0.10      # Asymmetric overlap ratio threshold

    # Temporal stability
    max_occlusion_frames: int = 20       # Keep fully-occluded vehicle alive this long
    temporal_alpha: float = 0.3          # EMA smoothing factor for score stability

    # Pruning gates (only these can truly remove a vehicle)
    confidence_floor: float = 0.20       # Below this YOLO conf → allow removal

    # Render-mode visibility thresholds (percentages)
    visibility_full: float = 80.0        # ≥ this → FULL render
    visibility_partial: float = 50.0     # ≥ this → PARTIAL render
    visibility_outline: float = 20.0     # ≥ this → OUTLINE render
    # Below visibility_outline and > 0   → POINT render
    # Exactly 0                          → PREDICTED render

    # Occlusion-score component weights (must sum to 1.0)
    w_visibility: float = 0.50
    w_distance: float = 0.20
    w_area: float = 0.15
    w_depth_rank: float = 0.15

    # Distance normalisation ceiling (metres) for scoring
    max_scoring_distance: float = 60.0

    # Minimum bbox area (pixels²) below which we consider the vehicle tiny
    min_bbox_area: int = 200


# ─── Per-Vehicle Output ──────────────────────────────────────────────────────

@dataclass
class VehicleOcclusionState:
    """Complete occlusion analysis output for a single tracked vehicle."""

    vehicle_id: int
    distance: float                          # metric distance in metres
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2) in image pixels
    visibility_score: float                  # 0–100 (100 = fully visible)
    occlusion_score: float                   # 0–100 (0 = fully visible)
    render_priority: int                     # lower = render first (nearest)
    render_mode: str                         # FULL | PARTIAL | OUTLINE | POINT | PREDICTED
    planner_active: bool                     # always True for valid tracked vehicles
    tracking_state: str                      # TRACKED | OCCLUDED | PREDICTED | LOST
    geometry: Any                            # passthrough VehicleGeometry reference
    occluded_by: List[int] = field(default_factory=list)
    frames_occluded: int = 0


# ─── Internal helpers ─────────────────────────────────────────────────────────

@dataclass
class _VehicleEntry:
    """Lightweight internal record used during per-frame processing."""
    vehicle_id: int
    bbox: Tuple[int, int, int, int]
    bbox_area: int
    distance: float
    confidence: float
    geometry: Any          # VehicleGeometry reference
    det_data: Dict         # raw detected_objects dict


# ─── OcclusionManager ────────────────────────────────────────────────────────

class OcclusionManager:
    """
    Computes occlusion / visibility for every tracked vehicle each frame.

    Usage::

        mgr = OcclusionManager()          # or OcclusionManager(OcclusionConfig(...))
        ...
        states = mgr.process(geometry_objects, detected_objects, frame_w, frame_h)
    """

    def __init__(self, config: Optional[OcclusionConfig] = None) -> None:
        self.config: OcclusionConfig = config or OcclusionConfig()

        # Temporal state  —  keyed by vehicle_id
        self._prev_visibility: Dict[int, float] = {}   # EMA-smoothed visibility
        self._frames_occluded: Dict[int, int] = {}      # consecutive occluded frames
        self._tracking_state: Dict[int, str] = {}        # last known tracking state
        self._seen_ids: set = set()                       # IDs seen this session

    # ── Static geometry helpers ───────────────────────────────────────────

    @staticmethod
    def compute_iou(a: Tuple[int, int, int, int],
                    b: Tuple[int, int, int, int]) -> float:
        """Standard Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def compute_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
        """Pixel area of a bounding box."""
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def compute_overlap_ratio(occluder: Tuple[int, int, int, int],
                              target: Tuple[int, int, int, int]) -> float:
        """Fraction of *target* bbox covered by *occluder*.

        Asymmetric: measures how much of target is hidden, not IoU.
        """
        ix1 = max(occluder[0], target[0])
        iy1 = max(occluder[1], target[1])
        ix2 = min(occluder[2], target[2])
        iy2 = min(occluder[3], target[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        target_area = (target[2] - target[0]) * (target[3] - target[1])
        return inter / target_area if target_area > 0 else 0.0

    @staticmethod
    def _intersection_box(a: Tuple[int, int, int, int],
                          b: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Return the intersection rectangle, or None if no overlap."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        if ix2 > ix1 and iy2 > iy1:
            return (ix1, iy1, ix2, iy2)
        return None

    # ── Core algorithms ───────────────────────────────────────────────────

    def _build_entries(
        self,
        geometry_objects: List[Any],
        detected_objects: List[Dict[str, Any]],
    ) -> List[_VehicleEntry]:
        """Merge geometry + detection data into a flat list of _VehicleEntry."""

        det_map: Dict[int, Dict] = {
            obj['track_id']: obj for obj in detected_objects
        }

        entries: List[_VehicleEntry] = []
        for vg in geometry_objects:
            det = det_map.get(vg.id)
            if det is None:
                continue  # no matching detection → skip (tracker lost it)

            bbox = tuple(det['bbox'])
            distance = 0.0
            if hasattr(vg, 'metric_estimate') and vg.metric_estimate is not None:
                distance = vg.metric_estimate.distance_m

            entries.append(_VehicleEntry(
                vehicle_id=vg.id,
                bbox=bbox,
                bbox_area=self.compute_bbox_area(bbox),
                distance=distance,
                confidence=det.get('conf', 0.0),
                geometry=vg,
                det_data=det,
            ))

        return entries

    def _depth_sort(self, entries: List[_VehicleEntry]) -> List[_VehicleEntry]:
        """Sort by metric distance ascending (nearest first)."""
        return sorted(entries, key=lambda e: e.distance)

    def _estimate_visibility(
        self,
        target: _VehicleEntry,
        occluders: List[_VehicleEntry],
    ) -> Tuple[float, List[int]]:
        """Estimate visible fraction of *target* considering all *occluders*.

        Uses a scanline-free approximation: accumulates intersection areas
        with nearer vehicles, then subtracts from total bbox area.  Where
        occluder regions themselves overlap we use the union of their
        intersections with the target to avoid double-subtraction.

        Returns:
            (visibility_score 0–100, list of occluder vehicle IDs)
        """
        if target.bbox_area <= 0:
            return 0.0, []

        occluder_ids: List[int] = []
        intersection_boxes: List[Tuple[int, int, int, int]] = []

        for occ in occluders:
            iou = self.compute_iou(occ.bbox, target.bbox)
            overlap = self.compute_overlap_ratio(occ.bbox, target.bbox)

            if iou > self.config.iou_threshold or overlap > self.config.overlap_threshold:
                inter = self._intersection_box(occ.bbox, target.bbox)
                if inter is not None:
                    intersection_boxes.append(inter)
                    occluder_ids.append(occ.vehicle_id)

        if not intersection_boxes:
            return 100.0, []

        # Compute union of all intersection rectangles with the target.
        # For typical vehicle counts (≤5 occluders per target) a simple
        # rasterisation on a downscaled 1-bit mask is fast and exact.
        tx1, ty1, tx2, ty2 = target.bbox
        tw = tx2 - tx1
        th = ty2 - ty1

        # Downsample for speed — 1 pixel per 4 real pixels
        SCALE = 4
        mw = max(1, tw // SCALE)
        mh = max(1, th // SCALE)
        mask = np.zeros((mh, mw), dtype=np.uint8)

        for bx1, by1, bx2, by2 in intersection_boxes:
            # Translate to target-local coords then scale
            lx1 = max(0, (bx1 - tx1) // SCALE)
            ly1 = max(0, (by1 - ty1) // SCALE)
            lx2 = min(mw, (bx2 - tx1 + SCALE - 1) // SCALE)
            ly2 = min(mh, (by2 - ty1 + SCALE - 1) // SCALE)
            mask[ly1:ly2, lx1:lx2] = 1

        occluded_fraction = float(np.count_nonzero(mask)) / float(mw * mh)
        visibility = max(0.0, min(100.0, (1.0 - occluded_fraction) * 100.0))
        return visibility, occluder_ids

    def _compute_occlusion_score(
        self,
        visibility: float,
        distance: float,
        bbox_area: int,
        depth_rank: int,
        total_vehicles: int,
    ) -> float:
        """Weighted composite occlusion score ∈ [0, 100]."""
        cfg = self.config

        # Component 1: inverse visibility (dominant)
        vis_component = 100.0 - visibility

        # Component 2: distance factor (further → higher score)
        dist_norm = min(1.0, distance / cfg.max_scoring_distance)
        dist_component = dist_norm * 100.0

        # Component 3: area factor (smaller bbox → more likely occluded)
        if bbox_area > 0:
            area_norm = 1.0 - min(1.0, bbox_area / 50000.0)  # 50k px² normaliser
        else:
            area_norm = 1.0
        area_component = area_norm * 100.0

        # Component 4: depth rank (further back in order → higher)
        rank_norm = depth_rank / max(1, total_vehicles - 1) if total_vehicles > 1 else 0.0
        rank_component = rank_norm * 100.0

        score = (
            cfg.w_visibility * vis_component
            + cfg.w_distance * dist_component
            + cfg.w_area * area_component
            + cfg.w_depth_rank * rank_component
        )
        return max(0.0, min(100.0, score))

    def _assign_render_mode(self, visibility: float) -> str:
        """Deterministic render-mode from visibility score."""
        cfg = self.config
        if visibility >= cfg.visibility_full:
            return "FULL"
        if visibility >= cfg.visibility_partial:
            return "PARTIAL"
        if visibility >= cfg.visibility_outline:
            return "OUTLINE"
        if visibility > 0.0:
            return "POINT"
        return "PREDICTED"

    def _update_temporal(self, vehicle_id: int, raw_visibility: float) -> Tuple[float, str, int]:
        """Apply EMA smoothing and manage temporal tracking state.

        Returns:
            (smoothed_visibility, tracking_state, frames_occluded)
        """
        alpha = self.config.temporal_alpha

        # EMA on visibility
        if vehicle_id in self._prev_visibility:
            smoothed = alpha * raw_visibility + (1.0 - alpha) * self._prev_visibility[vehicle_id]
        else:
            smoothed = raw_visibility
        self._prev_visibility[vehicle_id] = smoothed

        # Frames-occluded counter
        if smoothed < self.config.visibility_outline:
            self._frames_occluded[vehicle_id] = self._frames_occluded.get(vehicle_id, 0) + 1
        else:
            self._frames_occluded[vehicle_id] = 0

        frames_occ = self._frames_occluded[vehicle_id]

        # Tracking state machine
        if smoothed >= self.config.visibility_outline:
            state = "TRACKED"
        elif frames_occ <= self.config.max_occlusion_frames:
            state = "OCCLUDED"
        else:
            state = "PREDICTED"

        self._tracking_state[vehicle_id] = state
        self._seen_ids.add(vehicle_id)

        return smoothed, state, frames_occ

    def _cleanup_stale(self, active_ids: set) -> None:
        """Remove temporal state for vehicles no longer in the scene."""
        stale = self._seen_ids - active_ids
        for vid in stale:
            self._prev_visibility.pop(vid, None)
            self._frames_occluded.pop(vid, None)
            self._tracking_state.pop(vid, None)
        self._seen_ids -= stale

    # ── Main entry point ──────────────────────────────────────────────────

    def process(
        self,
        geometry_objects: List[Any],
        detected_objects: List[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
    ) -> List[VehicleOcclusionState]:
        """Compute occlusion state for every tracked vehicle in the frame.

        This is called once per frame.  It:
        1. Builds & depth-sorts the vehicle list.
        2. For each vehicle, estimates visibility against nearer vehicles.
        3. Computes occlusion score, render mode, temporal state.
        4. Sets planner_active = True for all (never hides from planner).

        Returns:
            List of VehicleOcclusionState, one per vehicle, sorted by
            render_priority (nearest first).
        """
        # Step 1 — merge and depth-sort
        entries = self._build_entries(geometry_objects, detected_objects)
        sorted_entries = self._depth_sort(entries)
        total = len(sorted_entries)

        active_ids: set = set()
        results: List[VehicleOcclusionState] = []

        for rank, entry in enumerate(sorted_entries):
            active_ids.add(entry.vehicle_id)

            # Step 2 — visibility estimation
            # Occluders are all entries that are *nearer* (lower index)
            nearer = sorted_entries[:rank]
            raw_vis, occluder_ids = self._estimate_visibility(entry, nearer)

            # Step 3 — temporal smoothing & state machine
            smoothed_vis, tracking_state, frames_occ = self._update_temporal(
                entry.vehicle_id, raw_vis
            )

            # Step 4 — occlusion score
            occ_score = self._compute_occlusion_score(
                smoothed_vis, entry.distance, entry.bbox_area, rank, total
            )

            # Step 5 — render mode & priority
            render_mode = self._assign_render_mode(smoothed_vis)
            render_priority = rank  # nearest = 0

            # Step 6 — planner always active
            planner_active = True

            results.append(VehicleOcclusionState(
                vehicle_id=entry.vehicle_id,
                distance=entry.distance,
                bbox=entry.bbox,
                visibility_score=round(smoothed_vis, 1),
                occlusion_score=round(occ_score, 1),
                render_priority=render_priority,
                render_mode=render_mode,
                planner_active=planner_active,
                tracking_state=tracking_state,
                geometry=entry.geometry,
                occluded_by=occluder_ids,
                frames_occluded=frames_occ,
            ))

        # Housekeeping — drop temporal state for vehicles no longer present
        self._cleanup_stale(active_ids)

        return results
