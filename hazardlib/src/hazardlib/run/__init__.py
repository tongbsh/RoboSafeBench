"""Run stage (R1, D16 c): ``hazardctl run`` drives the policy under test on frozen (or pilot) instances with
paired hazard-free controls; ``hazardctl report`` summarises the records. See :mod:`.runner` and :mod:`.report`."""
from .frames import (FRAME_INDEX_FILE, FrameRecorder, FrameSink, MemoryFrameSink, NpyFrameSink, TimedPolicy,
                     make_npy_sink)
from .inputs import RunInputError, RunInputs, load_formal, load_pilot
from .plan import CONTROL, HAZARD, PlannedEpisode, episode_horizon_s, plan_episodes, tau_s
from .report import ReportInputError, load_runs, render_markdown, summarize, write_report
from .runner import (EPISODES_FILE, RUN_META_FILE, PolicyInProbeSetError, RunDirectoryError, RunResult,
                     read_episode_records, run_stage)
from .settings import EpisodeSettings, ReportSettings, RunSettings
from .stats import proportion, proportion_interval, quartiles

__all__ = ["CONTROL", "EPISODES_FILE", "EpisodeSettings", "FRAME_INDEX_FILE", "FrameRecorder", "FrameSink", "HAZARD",
           "MemoryFrameSink", "NpyFrameSink", "PlannedEpisode", "PolicyInProbeSetError", "RUN_META_FILE",
           "ReportInputError", "ReportSettings", "RunDirectoryError", "RunInputError", "RunInputs", "RunResult",
           "RunSettings", "TimedPolicy", "episode_horizon_s", "load_formal", "load_pilot", "load_runs",
           "make_npy_sink", "plan_episodes", "proportion", "proportion_interval", "quartiles",
           "read_episode_records", "render_markdown", "run_stage", "summarize", "tau_s", "write_report"]
