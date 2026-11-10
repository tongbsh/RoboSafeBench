"""Run a cached OpenPI checkpoint on loopback, using an existing OpenPI checkout.

Explicit paths allow reuse of environments whose original interpreter symlink
has moved, without editing that environment or downloading dependencies.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    root = Path(args.repo)
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src"), args.site_packages]
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    print("Loading OpenPI modules", flush=True)
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    print("Loading cached checkpoint", flush=True)
    policy = create_trained_policy(get_config(args.config), args.checkpoint)
    from .seeded import PROTOCOL, RNG_KEY
    import jax
    class SeededPolicy:
        def infer(self, observation):
            observation = dict(observation)
            seed, request = observation.pop(RNG_KEY)
            policy._rng = jax.random.fold_in(jax.random.key(int(seed)), int(request))
            return policy.infer(observation)
    metadata = dict(policy.metadata, hazardlib_rng_protocol=PROTOCOL, checkpoint=args.checkpoint)
    server = WebsocketPolicyServer(policy=SeededPolicy(), host="127.0.0.1", port=args.port, metadata=metadata)
    faulthandler.cancel_dump_traceback_later()
    print("READY %s on 127.0.0.1:%d pid=%d" % (args.config, args.port, os.getpid()), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
