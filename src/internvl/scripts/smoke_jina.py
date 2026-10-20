"""Explicit real-model CPU smoke test; all model caches are temporary."""

import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _cache_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HF_MODULES_CACHE",
        "HF_XET_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "TRANSFORMERS_CACHE",
        "TMPDIR",
    ):
        directory = root / name.lower()
        directory.mkdir()
        env[name] = str(directory)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _terminate(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _run_worker(command: list[str]) -> int:
    """Wait for a child before removing its caches, including on interruption."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    worker = None
    try:
        for sig in previous:
            signal.signal(sig, _terminate)
        with TemporaryDirectory(prefix="internvl-smoke-") as directory:
            try:
                worker = subprocess.Popen(command, env=_cache_environment(Path(directory)))
                return worker.wait()
            finally:
                # A second signal must not interrupt child shutdown or cache removal.
                for sig in previous:
                    signal.signal(sig, signal.SIG_IGN)
                if worker is not None and worker.poll() is None:
                    worker.terminate()
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _worker() -> None:
    # Cache environment is established by the parent before any ML imports.
    import numpy as np
    from PIL import Image

    from internvl.embeddings import EmbeddingModel

    model = EmbeddingModel("jinaai/jina-clip-v2", device="cpu")
    with Image.new("RGB", (224, 224), color=(160, 80, 40)) as image:
        for modality, encode in (
            ("text", lambda: model.encode_text(["a bowl of rice"])),
            ("image", lambda: model.encode_images([image])),
        ):
            vector = encode()
            if (
                vector.shape != (1, 1024)
                or vector.dtype != np.float32
                or not np.isfinite(vector).all()
                or not np.any(vector)
            ):
                raise ValueError(f"Invalid {modality} embedding")
            print(f"{modality}: finite nonzero float32 {vector.shape}", flush=True)


def main() -> None:
    if sys.argv[1:] == ["--worker"]:
        _worker()
    elif sys.argv[1:]:
        raise SystemExit("Usage: python -m internvl.scripts.smoke_jina")
    else:
        raise SystemExit(
            _run_worker([sys.executable, "-m", "internvl.scripts.smoke_jina", "--worker"])
        )


if __name__ == "__main__":
    main()
