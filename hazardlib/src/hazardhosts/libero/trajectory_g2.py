"""G2 over the admitted instances of the all-demonstration analysis (development, pilot only).

For every layout already validated by ``trajectory_candidates.validate`` and every family whose selected
instance failed G1, run the privileged avoider (``avoider.NominalPathAvoider``) through the shared
``run_g2`` and store the outcome next to the G1 evidence (``<family>_g2.json``). The first episode of a
layout is a hazard-free run of the avoider's goal tracking, compared with the reference replay.

Labels follow ``gates.decision``: G2 passed -> ``evade_solvable``; G2 failed -> ``unsolvable_by_probes``
(not solvable by the probes we have; never a proof of impossibility).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hazardlib.config import default_config_dir, load_config
from hazardlib.driving import EpisodeSession
from hazardlib.gates.episodes import AvoiderController, run_g2
from hazardlib.gates.settings import GateSettings
from hazardlib.hashing import sha256_hex
from hazardlib.runtime import RuntimeConfig
from hazardlib.spec.instance import HazardInstance

from .avoider import NominalPath, NominalPathAvoider
from .host import LiberoHost
from .policies import DemoReplay
from .trajectory_analysis import save_json
from .trajectory_candidates import context


def evaluate(root, settings, avoider_settings, demos=None, families=None, suffix="g2"):
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    config = root / "config"
    avoider_hash = sha256_hex(avoider_settings)
    host = LiberoHost(config_dir=config)
    session = EpisodeSession(host, RuntimeConfig.load(config))
    reference = DemoReplay(host)
    gate_settings = GateSettings.load(config)
    geometry = {k: v.tolist() for k, v in np.load(root / "robot_geometry.npz").items()}
    dt = scene["control_dt_s"]
    try:
        for row in coverage["demonstrations"]:
            folder = root / "trajectories" / row["demo"]
            if (demos is not None and row["demo"] not in demos) or not (folder / "validation.json").exists():
                continue
            validation = json.loads((folder / "validation.json").read_text())
            pending = [f for f, r in sorted(validation["families"].items())
                       if r["selected"] and not r["selected"]["g1"]["passed"]
                       and (families is None or f in families)
                       and not (folder / ("%s_%s.json" % (f, suffix))).exists()]
            if not pending:
                continue
            arrays = dict(np.load(folder / "replay.npz"))
            tc = context(scene, row, arrays, geometry, validation["workspace"], identity)
            layout = settings["suite"] + "/" + row["demo"]
            nominal = NominalPath.record(session, scene["task_id"], layout, settings["seed"],
                                         len(arrays["time_s"]) * dt, avoider_settings["point_stride"])
            avoider = NominalPathAvoider(host, nominal, avoider_settings, lambda: session.phase(0))
            # Hazard-free goal tracking against the reference replay it was recorded from.
            free = session.run_episode(0, None, reference, scene["task_id"], layout, settings["seed"],
                                       len(arrays["time_s"]) * dt, controller=AvoiderController(avoider),
                                       control_id="g2-tracking:%s" % layout, stop_on_success=True)
            n = min(len(free.eef_pos), len(arrays["eef_pos"]))
            tracking = {"success": free.success, "success_t_s": free.success_t_s,
                        "max_eef_deviation_m": float(np.max(np.linalg.norm(
                            free.eef_pos[:n] - arrays["eef_pos"][:n], axis=1)))}
            print("TRACKING " + json.dumps(dict(tracking, demo=row["demo"])), flush=True)
            for family in pending:
                instance = HazardInstance.from_dict(validation["families"][family]["selected"]["instance"])
                avoider.arm(instance)
                run = run_g2(session, instance, reference, avoider, tc, gate_settings, settings["seed"], 0)
                label = "evade_solvable" if run.passed else "unsolvable_by_probes"
                save_json(folder / ("%s_%s.json" % (family, suffix)), {
                    "demo": row["demo"], "family": family, "instance_id": instance.instance_id,
                    "g2": run.to_dict(), "label": label, "tracking": tracking,
                    "avoider_settings_hash": avoider_hash, "avoider_settings": avoider_settings})
                episode = run.episodes[0]
                print("G2 " + json.dumps({"demo": row["demo"], "family": family, "passed": run.passed,
                                          "contacts": episode["official_contacts"], "success": episode["success"],
                                          "first_contact_t_s": episode["first_official_contact_t_s"],
                                          "t_hit_s": instance.nominal_hit_time_s,
                                          "avoider": episode["controller"].get("avoider_summary")}), flush=True)
    finally:
        host.close()


def summarize(root, settings, suffix="g2"):
    root = Path(root)
    counts = {f: {"wait_solvable": 0, "evade_solvable": 0, "unsolvable_by_probes": 0, "g2_pending": 0,
                  "no_admitted_candidate": 0} for f in settings["families"]}
    for path in sorted(root.glob("trajectories/demo_*/validation.json")):
        validation = json.loads(path.read_text())
        for family, record in validation["families"].items():
            if not record["selected"]:
                counts[family]["no_admitted_candidate"] += 1
            elif record["selected"]["g1"]["passed"]:
                counts[family]["wait_solvable"] += 1
            else:
                g2 = path.parent / ("%s_%s.json" % (family, suffix))
                counts[family][json.loads(g2.read_text())["label"] if g2.exists() else "g2_pending"] += 1
    save_json(root / ("%s_summary.json" % suffix), counts)
    print("SUMMARY " + json.dumps(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["evaluate", "summarize"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--demos")
    parser.add_argument("--families")
    parser.add_argument("--suffix", default="g2", help="output tag; use a new one for changed avoider settings")
    args = parser.parse_args()
    pilot = default_config_dir() / "pilot" / "libero"
    settings = load_config("trajectory_analysis", strict=True, config_dir=pilot)
    if args.stage == "evaluate":
        evaluate(args.out, settings, load_config("avoider", strict=True, config_dir=pilot),
                 None if args.demos is None else args.demos.split(","),
                 None if args.families is None else args.families.split(","), args.suffix)
    else:
        summarize(args.out, settings, args.suffix)


if __name__ == "__main__":
    main()
