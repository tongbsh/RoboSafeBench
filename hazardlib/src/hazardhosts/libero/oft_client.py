"""Client side of ``serve_oft``: a drop-in for ``OpenVLAOFTAdapter`` whose model runs in a separate process.

The DyniSafety adapter loads OpenVLA-OFT in-process, which the simulator interpreter cannot do (no ``prismatic``,
no OFT transformers fork); ``DyniPolicyBridge`` substitutes this class for it (``REMOTE_ADAPTERS`` in ``policies``)
and the same observation -> action-chunk contract is kept: primary + wrist image, the 8-d proprio state, the
instruction and the suite of the episode go out, the (chunk, 7) env-space actions (gripper already normalised and
inverted, as in the adapter) come back.
"""
import pickle
import socket
import struct

import numpy as np

PROTOCOL = "hazardlib-oft-v1"


def _send(stream, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<Q", len(data)) + data)
    stream.flush()


def _recv(stream):
    head = stream.read(8)
    if len(head) < 8:
        raise ConnectionError("OFT server closed the connection")
    (size,) = struct.unpack("<Q", head)
    data = stream.read(size)
    if len(data) < size:
        raise ConnectionError("OFT server closed the connection mid-message")
    return pickle.loads(data)


class OftRemoteAdapter:
    def __init__(self, cfg, host, port, timeout_s=600.0):
        self._cfg = cfg
        self._sock = socket.create_connection((host, port), timeout=timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stream = self._sock.makefile("rwb")
        hello = _recv(self._stream)
        self.metadata = hello.get("meta", {})
        if self.metadata.get("protocol") != PROTOCOL:
            self.close()
            raise RuntimeError("not a hazardlib OFT server on %s:%s; metadata=%r" % (host, port, self.metadata))
        self._suite = None

    @classmethod
    def from_config(cls, cfg):
        extra = cfg.extra
        return cls(cfg, extra["server_host"], int(extra["server_port"]))

    def reset(self, ctx):
        self._suite = ctx.suite
        if not self._suite:
            raise ValueError("OFT needs the suite of the episode (per-suite action statistics)")
        _send(self._stream, {"cmd": "reset"})
        _recv(self._stream)

    def act(self, obs):
        from dynisafety.policies.adapters._common import actions_to_chunk
        if self._suite is None:
            raise RuntimeError("act() before reset()")
        _send(self._stream, {"cmd": "infer", "main": np.asarray(obs.images["primary"]),
                             "wrist": np.asarray(obs.images["wrist"]), "state": np.asarray(obs.state, np.float64),
                             "prompt": obs.instruction, "suite": self._suite})
        reply = _recv(self._stream)
        if "actions" not in reply:
            raise RuntimeError("OFT server error: %s" % reply.get("error", reply))
        return actions_to_chunk(np.asarray(reply["actions"], np.float32), space=self._cfg.action_space)

    def close(self):
        try:
            _send(self._stream, {"cmd": "close"})
        except Exception:
            pass
        try:
            self._stream.close()
            self._sock.close()
        except Exception:
            pass
