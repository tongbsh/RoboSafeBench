# DyniSafety leaderboard (draft)

Draft submission format for sharing DyniSafety results. **No public submission
endpoint is published yet** (anonymous review). Replace the contact / URL
placeholders when a host is available — do not invent channels.

Scores are simulator-internal; they are **not** deployment safety certificates.
Do **not** rank absolute force across backends ([metrics.md](metrics.md),
[benchmark_design.md](benchmark_design.md)).

---

## 1. What to submit

Upload a single archive (`.zip` / `.tar.gz`) containing:

| Path in archive | Required | Description |
|-----------------|----------|-------------|
| `manifest.json` | yes | One JSON object describing the submission (schema below) |
| `runs.csv` | yes | Run-level metrics from `dynisafety report generate` |
| `coverage.csv` | yes | Coverage / gaps for the claimed grid |
| `table_main.md` | recommended | Paper-style main table |
| `resolved_configs/` | optional | One `resolved_config.yaml` per run (or hashes only in manifest) |
| `episodes/` | optional | Subset of `episodes.jsonl` for audit (can be large) |

Do **not** include checkpoints, videos, API keys, or absolute machine paths.

---

## 2. `manifest.json` schema (draft)

```json
{
  "schema_version": "1.0",
  "benchmark": "dynisafety",
  "suite": "paper_main",
  "submitter": "TODO_anonymous_or_team_id",
  "contact": "TODO",
  "git_sha": "<commit>",
  "git_dirty": false,
  "config_hashes": ["abc123def456"],
  "backends": ["libero", "simplerenv", "robotwin"],
  "notes": "SimplerEnv runtime.max_steps=120; force not cross-ranked"
}
```

`config_hashes` should match `config_hash` values from run manifests /
`EpisodeRecord` ([contracts.md](contracts.md) §6).

`paper_main` submissions list the three tabletop backends as in the example.
RoboCasa kitchen-nav is a fourth protocol and is **not** in `paper_main`; do not
silently add `"robocasa"` to a paper_main archive.

---

## 3. Ranking metrics (canonical IDs)

Primary leaderboard columns (display names in parentheses):

| ID | Display | Notes |
|----|---------|-------|
| `sr` | SR | Task success |
| `cr` | CR | Contact rate with disturbance |
| `sasr` | SASR | Safety-adjusted success |
| `topple_rate` | Topple | L1/L2 only; else n/a |
| `f_peak_mean_all` | FPeakMean(N) | **Per-backend only**; footnote no-contact → 0 N |

Optional diagnostics: `f_peak_mean_contact`, `safety_score`, injury bins — see
[metrics.md](metrics.md).

**Comparability:** rates may be compared across backends when defined; force
columns must be split by `force_source` / backend.

---

## 4. How to produce files

```bash
dynisafety sweep --suite paper_main …
dynisafety report generate \
  --results results/ \
  --out reports/ \
  --grid path/to/grid.yaml \
  --report-yaml path/to/report_settings.yaml \
  --metrics-yaml configs/base.yaml
```

Details: [reproducing_paper.md](reproducing_paper.md).

---

## 5. How to submit

| Step | Status |
|------|--------|
| Host URL | **TODO** (not published) |
| Email / form | **TODO** |
| License of submitted numbers | Same MIT protocol citation; respect third-party model licenses |

Until a host exists, keep archives private and cite [CITATION.cff](../../CITATION.cff)
/ README BibTeX (`TODO` authors under dual-blind).

---

## 6. Checklist

- [ ] Used `configs/suites/paper_main.yaml` (or clearly named subset)  
- [ ] Disclosed SimplerEnv **120-step** budget if reporting SimplerEnv SR  
- [ ] No cross-simulator force ranking  
- [ ] No PII / machine paths / weights in the archive  
- [ ] `coverage.csv` attached for claimed cells  
