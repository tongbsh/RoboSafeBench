"""Streaming MP4 FrameSink: records the exact RGB arrays passed to the policy."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from hazardlib.run.frames import safe_name


class Mp4FrameSink:
    def __init__(self, frames_dir, fps, crf, preset):
        self.root, self.fps, self.crf, self.preset = Path(frames_dir), fps, crf, preset
        self.episode = None
        self.streams = {}

    def open(self, episode_id, meta):
        if self.episode is not None:
            raise RuntimeError("video episode already open")
        self.episode = episode_id
        self.meta = dict(meta)
        digest = hashlib.sha256(episode_id.encode()).hexdigest()[:12]
        self.directory = self.root / (safe_name(episode_id)[:80] + "-" + digest)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.streams = {}

    def add(self, key, frame, t_s, phase):
        if self.episode is None:
            raise RuntimeError("video add without open")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("video frame must be HxWx3 uint8")
        height, width, _ = frame.shape
        if height % 2 or width % 2:
            raise ValueError("yuv420p requires even image dimensions")
        if key not in self.streams:
            path = self.directory / (safe_name(key) + ".mp4")
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", "%dx%d" % (width, height), "-r", str(self.fps),
                       "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", self.preset,
                       "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self.streams[key] = {"process": process, "path": path, "shape": frame.shape, "frames": []}
        stream = self.streams[key]
        if frame.shape != stream["shape"]:
            raise ValueError("video frame shape changed within episode")
        stream["process"].stdin.write(np.ascontiguousarray(frame).tobytes())
        stream["frames"].append({"t_s": float(t_s), "phase": phase})

    def close(self):
        if self.episode is None:
            raise RuntimeError("video close without open")
        results, cameras, errors = {}, {}, []
        for key, stream in self.streams.items():
            process = stream["process"]
            process.stdin.close()
            message = process.stderr.read().decode()
            code = process.wait()
            process.stderr.close()
            if code:
                errors.append("%s: ffmpeg exited %s: %s" % (key, code, message))
            results[key] = str(stream["path"])
            cameras[key] = {"path": stream["path"].name, "frames": stream["frames"]}
        index = self.directory / "frames.json"
        index.write_text(json.dumps({"episode_id": self.episode, "meta": self.meta,
                                     "fps": self.fps, "cameras": cameras}, indent=2))
        results["index"] = str(index)
        self.episode = None
        if errors:
            raise RuntimeError("; ".join(errors))
        return results
