"""Conformance tooling (01 §5). T04 provides the simulator-free ``FakeHost``; the suite itself is T05."""
from .fakehost import (ACTION_MODES, MUTATIONS, TABLE_NAME, FakeAction, FakeHost, FakeSphere, FakeTask,
                       FakeTaskObject, FakeWorldConfig)

__all__ = ["ACTION_MODES", "MUTATIONS", "TABLE_NAME", "FakeAction", "FakeHost", "FakeSphere", "FakeTask",
           "FakeTaskObject", "FakeWorldConfig"]
