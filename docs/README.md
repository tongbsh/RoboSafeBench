# Documentation

User-facing docs for DyniSafety (install, config, metrics, reproduction).

| Doc | Topic |
|-----|--------|
| [installation.md](installation.md) | Install backends, conda, Docker |
| [configuration.md](configuration.md) | YAML layout and CLI config |
| [contracts.md](contracts.md) | EpisodeRecord / results layout |
| [spawn_plan.md](spawn_plan.md) | `extra.spawn_plan` (L2 spawn diversity, field convention) |
| [metrics.md](metrics.md) | SR / CR / force / SASR |
| [behavior_diagnostics.md](behavior_diagnostics.md) | 行为诊断四轴（反应/停止/规避/恢复）— **已冻结（P5）** |
| [benchmark_design.md](benchmark_design.md) | Levels L0–L3 (tabletop + RoboCasa floor-nav) |
| [reproducing_paper.md](reproducing_paper.md) | Paper sweep + report |
| [adding_a_model.md](adding_a_model.md) | New policy adapter |
| [../examples/example_agent/README.md](../examples/example_agent/README.md) | Example VLA/WAM integration (local + HTTP) |
| [third_party.md](third_party.md) | Submodules and pins |
| [migration.md](migration.md) | Old commands → new CLI |
| [leaderboard.md](leaderboard.md) | Optional result submission |

Refactor handoff notes, work files, and historical drafts are **not** shipped
here; on developer machines they live under `old_cache/docs_internal/` (gitignored).
