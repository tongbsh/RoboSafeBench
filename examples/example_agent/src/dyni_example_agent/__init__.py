"""Example embodied-agent package for DyniSafety integration checks."""

from __future__ import annotations

from typing import Any

__all__ = ["ExampleAgentAdapter", "ExampleAgentHttpAdapter", "MockBackbone"]


def __getattr__(name: str) -> Any:
    if name == "ExampleAgentAdapter":
        from dyni_example_agent.local_adapter import ExampleAgentAdapter

        return ExampleAgentAdapter
    if name == "ExampleAgentHttpAdapter":
        from dyni_example_agent.http_adapter import ExampleAgentHttpAdapter

        return ExampleAgentHttpAdapter
    if name == "MockBackbone":
        from dyni_example_agent.mock_backbone import MockBackbone

        return MockBackbone
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
