# hazardlib

Simulator-agnostic core of the Hazard-as-Spec paradigm
(`docs/design/00_paradigm.md`, `01_integration_rules.md`). Implemented so far: data
structures and interfaces (T01) and the seconds-based kinematics and calibration math
(T02). Not yet: safety semantics, runtime, probes, gates, conformance tests (T03–T06).

## Layout

```
hazardlib/
  pyproject.toml                 numpy + pyyaml only; pytest as dev extra; Python >= 3.8
  src/hazardlib/                 core: must not import any simulator or dynisafety
    host.py                      HazardHost / OptionalHazardHost protocols, Tier, CapabilityError,
                                 Pose, RobotState, ContactObs, TimeState, Surface, HazardHandle, Action
    spec/                        HazardSpec, BenchmarkDescriptor, TaskContext, HazardInstance,
                                 Certificate, Manifest (+ PoseSequence)   to_dict / from_dict / validate
    kinematics/                  Path / SampledPath, min_jerk, free-fall conversions, straight/arc/reach
                                 paths, build_hit_path (nominal collision by construction), subdivide
    calibrate/                   estimate_t_stop, aggregate_t_stop, margin_sigma, staircase_update
    hashing.py                   canonical JSON -> sha256 (format rules in the module docstring)
    config.py                    YAML loading for configs/hazard/ with strict mode
    _records.py                  internal dataclass <-> dict machinery
  src/hazardhosts/<bench>/       empty adapter packages: libero simplerenv robotwin robocasa robodojo calvin
  tests/
../configs/hazard/               bd/calvin.yaml (draft), annotations/, families|intensity|gates|
                                 semantics|conformance.yaml (skeletons, every value PLACEHOLDER)
```

## Install and test

Install into each benchmark's own environment (rule R0), without dependency resolution:

```bash
pip install numpy pyyaml pytest          # if the environment lacks them
pip install -e hazardlib --no-deps
pytest hazardlib/tests -q                # run from the repository root
```

`configs/hazard/` is found by walking up from the installed source (editable install) or from
the working directory; set `HAZARDLIB_CONFIG_DIR` to point elsewhere.

## Python compatibility (3.8 through 3.11+, numpy 1.x and 2.x)

* Use `typing.List/Dict/Optional`, never `list[int]` or `X | Y` at runtime.
* No `match`, `dataclass(slots=True)`, `zip(strict=)`, `str.removeprefix`.
* No `typing_extensions`; `typing.Protocol` is available from 3.8.
* `tests/test_source_hygiene.py` enforces the above, the import ban (torch, scipy, mujoco,
  pybullet, sapien, isaac*, robosuite, dynisafety) and "no numeric default arguments".

## Conventions worth knowing

* **No numbers in code.** Thresholds and defaults live in `configs/hazard/*.yaml`. The T01 YAML files
  are structure with `PLACEHOLDER` leaves; a value is only real once someone replaces it. Numeric
  functions (kinematics, calibrate) take every number as an explicit argument; none has a default.
* **Paths clamp.** `Path.pos(t)` holds the end points outside the span and `vel(t)` is zero outside it.
* **Hit construction (decisions D1, D2, D4, D6 in `docs/design/00_paradigm.md` §13).** tau and `t_hit`
  follow the *first surface contact*; `build_hit_path` works on centroids, so the caller passes
  `hit_point = contact_point + standoff * n_hat` with `standoff = r_hazard + r_eef_contact` and records
  `standoff_m`. The slide/reach release velocity is the backward difference over the fixed
  `release_window_s` (YAML `release_velocity_window_s`), never the engine hook step. `fall` is released at
  rest at the onset with the host's measured gravity. A `min_jerk` reach with `overshoot_m = 0` ends at rest
  and must not carry severity. Details: the `kinematics/hit.py` docstring.
* **`unknown` and `PLACEHOLDER` are not values.** `load_*(..., strict=True)` refuses any file containing
  either and lists every path. `strict=False` loads them so drafts can be reviewed.
* **`BenchmarkDescriptor.tier()` never guesses.** If the Tier depends on an `unknown` field it raises
  `TierUndeterminedError` naming the fields.
* **Records are strict:** unexpected keys, missing required keys, wrong types and non-finite numbers are errors
  with a dotted path (`HazardSpec.agent: half_extents_m[1] ...`).
* **YAML 1.1 gotchas (PyYAML):** quote versions and commit ids (`"3.8"`, `"797142c"`); write exponents
  with a dot (`1.0e-3`); duplicate keys are rejected by our loader.
* **Hashing:** `Record.hash()` = sha256 of canonical JSON of `to_dict()`; key order is irrelevant, floats are
  compared at 12 significant digits. A pinned golden digest in `tests/test_hashing.py` guards the format.

## Conformance suite (C1–C13, T05)

Checks C1–C13 of `01_integration_rules.md` §5 live in `src/hazardlib/conformance/tests/test_cNN_*.py`
(one check per file) and run against any host registered under a name:

```bash
python -m hazardlib.conformance --host fake --out report.json      # exit 0 pass, 1 fail, 2 unverified
pytest hazardlib/src/hazardlib/conformance/tests --host=fake        # same checks through pytest
```

A host module calls `register_host(name, factory)`; the factory returns a `ConformanceTarget` (the host, its
scenario geometry, a scripted motion, optionally its instance builder, and the fault injector
`inject_engine_fault` that C11 needs). `--host <name>` imports `hazardhosts.<name>` unless `--host-module` names
the module. Tolerances and their derivations are in `configs/hazard/conformance.yaml`; the built-in host `fake` is
FakeHost in the world of `configs/hazard/fakehost.yaml`. `tests/test_mutants.py` proves that each FakeHost mutation
is caught.

Statuses follow 01 §5: a host lacking a capability a check needs (`CapabilityError`) fails that check; `error` is
kept for a crash of the suite itself. Check-specific rules from D13 h:

* C3 crosses a robot body within half a physics step when the BD declares `hazard_ccd: true`, otherwise at half
  the R1 limit speed (the BD has no maximum-speed field).
* C5 requires zero linear and angular hazard damping; the landing-time tolerance is a number of physics steps of
  the BD's `physics_dt_s`.
* C11 forces an engine fault through the adapter's `inject_engine_fault(env, hazard)`; without it C11 fails.
  FakeHost has no real engine faults, so on FakeHost C11 verifies only the mapping path.
* C12 runs on every host with `num_envs > 1`.
* On a host declaring `same_seed_repro: statistical`, C2 / C7 / C12 compare against C8's same-seed bounds, given
  per observation kind (state vectors; images, every camera on its own). While a bound is PLACEHOLDER the check is
  `unverified` and the report carries the measured magnitude; the host's H task sets the bound.

The report's `config_source` names the config directory relative to the repository root (`configs/hazard`), or
`external:<directory name>` outside it, never a machine path; `config_hashes` pins the content. Freezing needs a
`pass` report run without `probe_runtime` / `probe_calibration` fallbacks (D13 a).
