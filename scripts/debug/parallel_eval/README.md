# Parallel-eval P0 alignment

N=1 seeded alignment of the HTTP `/act` path (`xvla_official`) against
`dynisafety parallel-eval` (`xvla_official_parallel`). Do not use this as the
paper entrypoint — `scripts/eval/` is unchanged.

## `--seed` vs `rng_seed` (do not confuse these)

| Flag / key | What it controls | What it does **not** control |
|---|---|---|
| `--seed` → `runtime.seed` | Expert-check cursor / `EpisodeRecord.seed` / seed-warm `run_seed` | Per-slot flow-matching noise `x1` |
| `extra.parallel_eval.rng_seed` (`--rng-seed`) | `mix_rng_key` → `ObsMsg.rng_key` → Infer `slot_noise` (plan 0.13) | Expert-check / demo seed |

YAML `rng_seed` is required (R4: no Python default). `xvla_official_parallel.yaml` currently sets `0`. Passing `--seed 7` on P0 **does not** change that `0`. If you need a specific x1 stream, pass `--rng-seed N` (maps to `--set extra.parallel_eval.rng_seed=N`) or edit the YAML. Startup logs print both values so a mismatch is visible.

P0: keep `--seed` identical to the HTTP baseline (`p0_align.sh` default `7`). Leave `rng_seed` at the YAML value unless you are deliberately testing x1; HTTP `/act` cannot inject `x1`, so action-chunk diffs may still be the 0.13 noise issue rather than a seed bug.

## Why `prompt_mode=task_name`

`select_instruction` in `official` / `random` mode calls global
`np.random.choice`. The HTTP rollout and the Infer/Env split do not share a
process-wide RNG stream, so the instruction string can differ even when seeds
match. `task_name` (and YAML `default`, which aliases to it) is deterministic.
`p0_align.sh` forces `--set extra.robotwin.prompt_mode=task_name` on both paths.

## How to run

Needs two conda envs (`XVLA` for Infer, `internvla_a1` for Env / HTTP eval),
GPU, and a live X-VLA HTTP server for the baseline.

```bash
# terminal 1 — HTTP serve (xvla_official)
bash scripts/debug/robotwin/launch_xvla_official_robotwin_serve.sh

# terminal 2
bash scripts/debug/parallel_eval/p0_align.sh --task 21 --level 0 --episodes 1 --seed 7
# optional: --rng-seed N  (only if you want to override YAML x1 seed)
```

Overrides: `--host` / `--port` (default `127.0.0.1:8010`), `--out`,
`--skip-http` (warm + parallel only), `--no-wait`, `--rng-seed`.

The script:

1. **Seed-warm** — serial `_seed_cursor` walk. Writes
   `(episode_index, start_seed, accepted_seed)` and fills the expert-check
   cache. Parallel jobs pass **start_seed** into `begin` so the cache key hits
   (plan 0.7).
2. **HTTP baseline** — `dynisafety eval --model xvla_official`.
3. **Parallel** — `dynisafety parallel-eval --model xvla_official_parallel`
   with `extra.parallel_eval.enabled=true`, `num_envs=1`,
   `deterministic_rng=true`. `--seed` matches the HTTP job seed;
   `rng_seed` stays on the YAML / `--rng-seed` value.
4. **Compare** the two `episodes.jsonl` (+ diagnostics traces).

## What each gate checks

| Gate | Where |
|---|---|
| Action chunk | Per-control-step `action` on the diagnostics trace (`traces/epXXXX.jsonl.gz`). These are the applied rows after official 20-D → EE decode, not raw Infer tensors. HTTP cannot inject `x1`; a miss here may be the 0.13 noise issue, not a seed bug. |
| `accepted_seed` | `extra.accepted_seed` |
| `success` / `termination` | EpisodeRecord fields |
| Sim-step contact clock | `metrics.first_contact_step` (`first_contact_step` on the record) |
| Control-step first contact | `extra.first_contact_control_step` |
| Control-step contact count | `extra.contact_control_steps` |

P0 items 5–6 (batch invariance / B-independent `x1`) are unit-tested in
`tests/test_parallel_eval_infer.py`, not this script.

## Manual pieces (same as the script)

```bash
# 1. warm (must run inside the Env conda — needs sapien)
conda run --no-capture-output -n internvla_a1 python -m dynisafety.cli parallel-eval \
  --backend robotwin --model xvla_official_parallel --suite main20 --level 0 \
  --task 21 --episodes 1 --seed 7 \
  --seed-warm artifacts/debug/parallel_eval/accepted_seeds.json \
  --warm-only \
  --set extra.robotwin.prompt_mode=task_name

# 2. HTTP baseline (serve already up)
HOST=127.0.0.1 PORT=8010 conda run --no-capture-output -n internvla_a1 \
  python -m dynisafety.cli eval \
  --backend robotwin --model xvla_official --suite main20 --level 0 \
  --task 21 --episodes 1 --seed 7 \
  --set extra.robotwin.prompt_mode=task_name

# 3. parallel
conda run --no-capture-output -n internvla_a1 python -m dynisafety.cli parallel-eval \
  --backend robotwin --model xvla_official_parallel --suite main20 --level 0 \
  --task 21 --episodes 1 --seed 7 \
  --seed-warm artifacts/debug/parallel_eval/accepted_seeds.json \
  --set extra.robotwin.prompt_mode=task_name \
  --set extra.parallel_eval.enabled=true \
  --set extra.parallel_eval.num_envs=1 \
  --set extra.parallel_eval.deterministic_rng=true
  # add --rng-seed N only to override YAML extra.parallel_eval.rng_seed
```
