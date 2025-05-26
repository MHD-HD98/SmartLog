import subprocess
import numpy as np
from typing import Optional

_ffmpeg_stream_instance = None  # Internal singleton


def get_stream_instance():
    global _ffmpeg_stream_instance
    return _ffmpeg_stream_instance


def set_stream_instance(stream):
    global _ffmpeg_stream_instance
    _ffmpeg_stream_instance = stream


class FFmpegStream:
    def __init__(
        self, source: str, width: int = 1280, height: int = 720, fps: int = 20
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe: Optional[subprocess.Popen] = None

    def start(self):
        if self.pipe and self.pipe.poll() is None:
            return  # Already running

        command = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.source,
            "-r",
            str(self.fps),  # Set output FPS
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-an",
            "-sn",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-vf",
            f"scale={self.width}:{self.height}",
            "-",
        ]
        self.pipe = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.pipe or self.pipe.poll() is not None:
            return None

        frame_size = self.width * self.height * 3
        raw_frame = self.pipe.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            return None

        frame = (
            np.frombuffer(raw_frame, dtype=np.uint8)
            .reshape((self.height, self.width, 3))
            .copy()
        )
        return frame

    def stop(self):
        if self.pipe:
            self.pipe.kill()
            self.pipe = None

    def is_running(self):
        return self.pipe and self.pipe.poll() is None
