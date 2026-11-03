"""Own an Ollama client and the server process used by one command."""

import os
import shutil
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from ollama import Client


class OllamaSession:
    """Provide a client while managing only the server this session starts."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Construct a client from settings checked by ``validate_ollama_config``.

        The context manager starts the server on entry and stops it on exit.
        """
        startup_timeout = config["startup_timeout"]
        request_timeout = config["request_timeout"]
        self.binary_path = config["binary_path"]
        self.host = config["host"]
        if "://" not in self.host:
            self.host = "http://" + self.host
        self.startup_timeout = startup_timeout
        self.client = Client(host=self.host, timeout=request_timeout)
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Self:
        self.start_server()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Terminate the owned server, killing it if the shutdown wait expires."""
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        finally:
            self._process = None

    def _is_ready(self, timeout: float) -> bool:
        """Return False for transient connection failures; reject invalid responses."""
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as probe:
                response = probe.get(self.host.rstrip("/") + "/api/tags")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                    raise RuntimeError("Endpoint did not return an Ollama model list")
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        return True

    def _validate_binary(self) -> str:
        """Resolve the configured executable without installing or running it."""
        guidance = (
            "Install Ollama or set ollama.binary_path to an existing executable. "
            "See README.md: Ollama setup."
        )
        if not isinstance(self.binary_path, str) or not self.binary_path.strip():
            raise ValueError(f"ollama.binary_path must be a nonempty string. {guidance}")
        binary = self.binary_path
        if "/" not in binary and not binary.startswith("~"):
            resolved = shutil.which(binary)
            if resolved is None:
                raise FileNotFoundError(
                    f"ollama.binary_path executable not found on PATH: {binary}. {guidance}"
                )
            binary = resolved
        path = Path(binary).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Ollama executable not found: {path}. {guidance}")
        if not path.is_file():
            raise ValueError(f"ollama.binary_path is not a file: {path}. {guidance}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Ollama executable lacks execute permission: {path}. {guidance}")
        return str(path)

    def start_server(self) -> None:
        """Launch Ollama, rejecting an endpoint already served by another instance."""
        deadline = time.monotonic() + self.startup_timeout
        try:
            if self._process is None:
                binary_path = self._validate_binary()
                # If self._process was not set, but an Ollama instance is already running
                # on this port, abort.
                if self._is_ready(timeout=min(1, self.startup_timeout)):
                    raise RuntimeError(f"An Ollama instance is already bound to {self.host}")
                env = os.environ.copy()
                env["OLLAMA_HOST"] = self.host
                # Force Ollama subprocess to inherit model environment.
                try:
                    self._process = subprocess.Popen([binary_path, "serve"], env=env)
                except OSError as exc:
                    raise RuntimeError(
                        f"Could not launch Ollama executable {binary_path}: {exc}. "
                        "Check the installation at ollama.binary_path. "
                        "See README.md: Ollama setup."
                    ) from exc

            while True:
                if self._process.poll() is not None:
                    raise RuntimeError("Ollama exited during startup.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Ollama failed to start at {self.host} in timeout period {self.startup_timeout}."
                    )
                if self._is_ready(timeout=min(1, remaining)):
                    if self._process.poll() is not None:
                        raise RuntimeError("Ollama exited during startup; check its stderr")
                    return
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        except BaseException:
            self.close()
            raise

    def download_model(self, model_name: str) -> None:
        """Pull a requested model with visible progress from the owned server."""
        for progress in self.client.pull(model=model_name, stream=True):
            message = progress.status
            if progress.total is not None and progress.completed is not None:
                message = f"{message}: {progress.completed}/{progress.total} bytes"
            print(message, flush=True)
