"""Serve OpenVLA-OFT on loopback for the cell evaluation (L6), from an existing OFT inference environment.

OFT is an in-process policy for DyniSafety (``transport: local``) and needs ``prismatic`` / the OFT transformers
fork / TensorFlow, none of which the simulator interpreter has. This script runs the *official* per-sample inference
path (``experiments.robot.openvla_utils``: ``get_vla`` / ``get_processor`` / ``get_proprio_projector`` /
``get_action_head`` / ``get_vla_action``; L1-regression head, two images + proprio, 8-step chunk, gripper
normalise+invert) in that environment and answers the simulator workers over a loopback socket (``oft_client``).
Nothing is installed or edited: pass an existing interpreter and OFT checkout, e.g.

    <venv-openvla>/bin/python -m hazardhosts.libero.serve_oft --repo <third_party/openvla-oft> \\
        --checkpoint <openvla-7b-oft-finetuned-libero-spatial-object-goal-10> \\
        --view-root <scratch> --port 18030 --gpu 0

The official loader rewrites ``config.json`` and copies the modeling code into the checkpoint directory, so the
checkpoint is never loaded directly: a *view* directory (symlinks to the weights, copies of the small files) is built
under ``--view-root`` and loaded instead.

The checkpoint is the combined spatial/object/goal/10 fine-tune: its ``dataset_statistics.json`` holds one
``<suite>_no_noops`` entry per suite, selected per request through the suite of the episode (OpenVLA-OFT is
deterministic: no sampling RNG, so the seeded OpenPI protocol is not involved).

Wire format: 8-byte little-endian length + pickle (loopback only, both ends are ours). The server first sends
``{"meta": {...}}``; a request ``{"cmd": "infer", "main", "wrist", "state", "prompt", "suite"}`` is answered with
``{"actions": (chunk, 7) float32}`` or ``{"error": traceback}``; ``{"cmd": "reset"}`` is acknowledged.
"""
import argparse
import json
import os
import pickle
import shutil
import socketserver
import struct
import sys
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

PROTOCOL = "hazardlib-oft-v1"


def send_message(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def recv_message(stream):
    head = stream.read(8)
    if len(head) < 8:
        return None
    (size,) = struct.unpack("<Q", head)
    data = stream.read(size)
    if len(data) < size:
        return None
    return pickle.loads(data)


def make_view(source: Path, view: Path) -> Path:
    """Symlink the weights, copy the small files (the official loader writes into the directory it loads)."""
    if view.exists():
        shutil.rmtree(str(view))
    view.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix in (".json", ".py", ".txt", ".md") and entry.stat().st_size < 50000000:
            shutil.copy2(str(entry), str(view / entry.name))
        else:
            os.symlink(str(entry.resolve()), str(view / entry.name))
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="openvla-oft checkout (experiments.robot / prismatic)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--view-root", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, args.repo)
    sys.argv = [sys.argv[0], "libero"]       # prismatic.vla.constants picks the LIBERO constants from argv
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    from experiments.robot import openvla_utils as ou
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

    source = Path(args.checkpoint)
    view = make_view(source, Path(args.view_root) / ("oft-%s-%d" % (source.name, args.port)))
    cfg = SimpleNamespace(
        pretrained_checkpoint=str(view), model_family="openvla_oft", load_in_8bit=False, load_in_4bit=False,
        use_l1_regression=True, use_diffusion=False, use_film=False, num_images_in_input=2, use_proprio=True,
        center_crop=True, num_open_loop_steps=8, lora_rank=32, unnorm_key="libero_spatial", task_suite_name="libero_spatial",
        num_diffusion_steps_train=50, num_diffusion_steps_inference=50)
    print("Loading OFT checkpoint view %s" % view, flush=True)
    vla = ou.get_vla(cfg)
    processor = ou.get_processor(cfg)
    proprio_projector = ou.get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    action_head = ou.get_action_head(cfg, vla.llm_dim)
    stats = vla.norm_stats

    def unnorm_key(suite: str) -> str:
        for key in (suite, suite + "_no_noops"):
            if key in stats:
                return key
        raise KeyError("no norm_stats for suite %r (have %s)" % (suite, sorted(stats)))

    lock = threading.Lock()

    def infer(request):
        key = unnorm_key(request["suite"])
        observation = {"full_image": ou.resize_image_for_policy(request["main"], 224),
                       "wrist_image": ou.resize_image_for_policy(request["wrist"], 224),
                       "state": np.asarray(request["state"], np.float64)}
        with lock:
            cfg.unnorm_key = key
            actions = ou.get_vla_action(cfg, vla, processor, observation, request["prompt"], action_head=action_head,
                                        proprio_projector=proprio_projector, use_film=False)
        out = []
        for action in actions[:cfg.num_open_loop_steps]:
            out.append(invert_gripper_action(normalize_gripper_action(np.asarray(action, np.float64).copy(),
                                                                       binarize=True)))
        return np.stack(out).astype(np.float32)

    meta = {"protocol": PROTOCOL, "checkpoint": str(source), "unnorm_keys": sorted(stats), "chunk": cfg.num_open_loop_steps,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            send_message(self.wfile, {"meta": meta})
            while True:
                request = recv_message(self.rfile)
                if request is None or request.get("cmd") == "close":
                    return
                try:
                    if request["cmd"] == "reset":
                        send_message(self.wfile, {"ok": True})
                    else:
                        send_message(self.wfile, {"actions": infer(request)})
                except Exception as exc:     # reported to the worker instead of dying silently
                    send_message(self.wfile, {"error": traceback.format_exc() + repr(exc)})

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server(("127.0.0.1", args.port), Handler)
    print("READY openvla_oft %s on 127.0.0.1:%d pid=%d" % (json.dumps(meta["unnorm_keys"]), args.port, os.getpid()),
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
