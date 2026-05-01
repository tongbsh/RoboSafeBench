#!/usr/bin/env python
"""Thin LingBot-VA entrypoint for DyniSafety-RoboTwin."""

from __future__ import annotations

from main_inference import run_policy_entrypoint


if __name__ == "__main__":
    run_policy_entrypoint("lingbot_va", default_prompt_mode="official")
