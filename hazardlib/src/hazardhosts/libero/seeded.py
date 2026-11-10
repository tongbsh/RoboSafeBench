"""Explicit per-episode, per-request RNG protocol for paired OpenPI rollouts."""
PROTOCOL = "episode-seed-request-v1"
RNG_KEY = "_hazardlib_rng"


class SeededTransport:
    def __init__(self, transport):
        metadata = getattr(getattr(transport, "_codec", None), "server_metadata", {})
        if metadata.get("hazardlib_rng_protocol") != PROTOCOL:
            transport.close()
            raise RuntimeError("paired OpenPI runs require the hazardlib seeded server; metadata=%r" % metadata)
        self.transport = transport
        self.seed = None
        self.index = 0

    def set_seed(self, seed):
        self.seed, self.index = int(seed), 0

    def reset(self, payload):
        self.index = 0
        return self.transport.reset(payload)

    def infer(self, payload):
        if self.seed is None:
            raise RuntimeError("policy RNG seed was not reset for the episode")
        message = dict(payload)
        message[RNG_KEY] = [self.seed, self.index]
        self.index += 1
        return self.transport.infer(message)

    def close(self):
        return self.transport.close()
