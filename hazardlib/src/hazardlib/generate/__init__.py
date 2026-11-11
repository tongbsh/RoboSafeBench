"""The perturbation paradigm's generator (D20): hazard proposals from two input variables.

V1 (successful execution trajectories) and V2 (the 3D scene layout) are the only per-benchmark inputs
(:mod:`.inputs`); everything else (families, 3D entry directions, hit-point choice and trajectory-intersection
scoring, Sigma levels, validity) is the paradigm (:mod:`.propose`, :mod:`.clearance`, :mod:`.select`) and
reads its numbers from YAML (:mod:`.settings`). numpy only; no simulator.
"""
from .clearance import layout_probes, playback_path, rejection_reason, support_height, sweep
from .geometry import box_overlap_depth, points_box_distance, points_boxes_distance, quat_to_mat
from .inputs import LayoutBody, SceneLayout, Segment, Trajectory, TrajectorySet
from .propose import Cell, CoverageIndex, propose, tau_for_sigma, target_key
from .select import cell_statistics, generation_table, select_admitted
from .settings import EntrySettings, GenerateSettings, cell_id, elevation_tag

__all__ = ["Cell", "CoverageIndex", "EntrySettings", "GenerateSettings", "LayoutBody", "SceneLayout", "Segment",
           "Trajectory", "TrajectorySet", "box_overlap_depth", "cell_id", "cell_statistics", "elevation_tag",
           "generation_table", "layout_probes", "playback_path", "points_box_distance", "points_boxes_distance",
           "propose", "quat_to_mat", "rejection_reason", "select_admitted", "support_height", "sweep",
           "target_key", "tau_for_sigma"]
