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
  `hit_point` for `build_hit_path` is the hazard *centroid* target; offsets and contact standoffs are
  added by the caller (see the `kinematics/hit.py` docstring).
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
