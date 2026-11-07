"""Solvability gates G0 / G1 / G2 and the certificate decision table (T06, paradigm §8)."""
from .decision import (MAIN_SCORE_LABELS, decide_label, filter_main_score, in_main_score, label_of,
                       make_certificate)
from .episodes import (AvoiderController, BaselineResult, GateRun, OracleController, PrivilegedAvoider,
                       nominal_success_time_s, run_baseline, run_g1, run_g2, run_nominal_hit)
from .exposure import annotate_exposure, exposure_low, load_low_exposure_radius_m, low_exposure_radius_m
from .g0 import G0_REASONS, G0Result, run_g0
from .settings import EpisodeGateSettings, G0Settings, GateSettings
from .stage import GateOutput, gate_instances

__all__ = ["AvoiderController", "BaselineResult", "EpisodeGateSettings", "G0Result", "G0Settings", "G0_REASONS",
           "GateOutput", "GateRun", "GateSettings", "MAIN_SCORE_LABELS", "OracleController", "PrivilegedAvoider",
           "annotate_exposure", "decide_label", "exposure_low", "filter_main_score", "gate_instances",
           "in_main_score", "label_of", "load_low_exposure_radius_m", "low_exposure_radius_m", "make_certificate",
           "nominal_success_time_s", "run_baseline", "run_g0", "run_g1", "run_g2", "run_nominal_hit"]
