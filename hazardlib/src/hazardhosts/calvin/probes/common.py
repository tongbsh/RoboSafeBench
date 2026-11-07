"""Shared helpers of the CALVIN probes: build the hooked env on a pinned EGL device, name bodies and links."""
import argparse
import os
import subprocess
from typing import Any, Dict, List, Tuple

DATASET_ENV = "CALVIN_DATASET_DIR"
EGL_ENV = "CALVIN_EGL_DEVICE"


def probe_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-dir", default=os.environ.get(DATASET_ENV),
                        help="calvin_debug_dataset directory (default: $%s)" % DATASET_ENV)
    parser.add_argument("--egl-device", type=int, default=None,
                        help="EGL device index (default: $%s)" % EGL_ENV)
    parser.add_argument("--cameras", default="static,gripper", help="comma-separated camera names")
    return parser


def build_env(args: Any) -> Any:
    from hazardhosts.calvin.env import make_env, pin_egl_device
    egl = args.egl_device if args.egl_device is not None else os.environ.get(EGL_ENV)
    if egl is None:
        raise SystemExit("set --egl-device or $%s (see probes/egl_devices.py)" % EGL_ENV)
    if not args.dataset_dir:
        raise SystemExit("set --dataset-dir or $%s" % DATASET_ENV)
    pin_egl_device(int(egl))
    cameras = [c for c in args.cameras.split(",") if c] if args.cameras else None
    return make_env(os.path.join(args.dataset_dir, "validation"), cameras)


def body_names(env: Any) -> Dict[int, str]:
    """uid -> name: ``robot``, the scene's fixed / movable object names, else the URDF body name."""
    p, cid = env.p, env.cid
    names = {env.robot.robot_uid: "robot"}
    for obj in env.scene.get_objects():
        names[obj.uid] = obj.name
    for i in range(p.getNumBodies(physicsClientId=cid)):
        uid = p.getBodyUniqueId(i, physicsClientId=cid)
        if uid not in names:
            names[uid] = p.getBodyInfo(uid, physicsClientId=cid)[1].decode() or "body%d" % uid
    return names


def link_names(env: Any, uid: int) -> Dict[int, str]:
    p, cid = env.p, env.cid
    out = {-1: p.getBodyInfo(uid, physicsClientId=cid)[0].decode()}
    for j in range(p.getNumJoints(uid, physicsClientId=cid)):
        out[j] = p.getJointInfo(uid, j, physicsClientId=cid)[12].decode()
    return out


def collision_links(env: Any) -> List[Tuple[int, int, str, str, List[Any]]]:
    """``(uid, link, body name, link name, collision shapes)`` of every link with a collision shape."""
    p, cid = env.p, env.cid
    rows = []
    for uid, name in sorted(body_names(env).items()):
        links = link_names(env, uid)
        for link in sorted(links):
            shapes = p.getCollisionShapeData(uid, link, physicsClientId=cid)
            if shapes:
                rows.append((uid, link, name, links[link], list(shapes)))
    return rows


def nvidia_smi_processes(pid: int) -> str:
    """The ``nvidia-smi`` process table lines of ``pid`` plus the GPU index / bus id table."""
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                          stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    table = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, check=True).stdout.decode()
    lines = [line for line in table.splitlines() if " %d " % pid in line]
    return "GPUs:\n%s\nprocess %d in nvidia-smi:\n%s" % (gpus, pid, "\n".join(lines) or "(none)")
