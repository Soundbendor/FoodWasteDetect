import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import httpx
import ollama
from ollama import ChatResponse, Client
from pydantic import BaseModel

from data_wrappers.dataset import Dataset


def descriptor_prompt(cleaned_item: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Can you provide a sentence describing the food {cleaned_item}? "
            "Please be as descriptive as possible, focusing on the visual "
            f"characteristics of {cleaned_item} and explicitly mentioning "
            f"{cleaned_item} in your description."
        ),
    }


class FoodDescriptor(BaseModel):
    food_class: str
    description: str


class DescriberLLM:
    def __init__(self, config: dict):
        """Read required settings from the experiment's ollama configuration."""
        startup_timeout = config["startup_timeout"]
        request_timeout = config["request_timeout"]
        if startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self.model = config["llm"]
        self.binary_path = config["binary_path"]
        self.host = config["host"]
        if "://" not in self.host:
            self.host = "http://" + self.host
        self.startup_timeout = startup_timeout
        self.client = Client(host=self.host, timeout=request_timeout)
        self._process: subprocess.Popen | None = None

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Safely shut down the Ollama server launched by this instance."""
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
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("models"), list
                ):
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
            raise ValueError(
                f"ollama.binary_path must be a nonempty string. {guidance}"
            )
        path = Path(self.binary_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Ollama executable not found: {path}. {guidance}")
        if not path.is_file():
            raise ValueError(f"ollama.binary_path is not a file: {path}. {guidance}")
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"Ollama executable lacks execute permission: {path}. {guidance}"
            )
        return str(path)

    def start_server(self):
        """Launch Ollama, rejecting an endpoint already served by another instance."""
        deadline = time.monotonic() + self.startup_timeout
        try:
            if self._process is None:
                binary_path = self._validate_binary()
                # If self._process was not set, but an Ollama instance is already running
                # on this port, abort.
                if self._is_ready(timeout=min(1, self.startup_timeout)):
                    raise RuntimeError(
                        f"An Ollama instance is already bound to {self.host}"
                    )
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
                        raise RuntimeError(
                            "Ollama exited during startup; check its stderr"
                        )
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

    def _describe_class(self, label: str, prompt: dict[str, str]) -> FoodDescriptor:
        """Request and validate one description, retaining the supplied class label."""
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=[prompt],
            format=FoodDescriptor.model_json_schema(),
        )
        output = FoodDescriptor.model_validate_json(response.message.content)
        output.food_class = label
        output.description = output.description.replace("*", "")
        return output

    def generate_descriptors(self, ds: Dataset, save_path: str) -> str:
        """Given a dataset, provides a list of descriptions of each class of that dataset"""
        ds_descriptors = defaultdict(list)
        for item in ds.get_class_list():
            prompt = descriptor_prompt(item.replace("food", "").strip())
            for _ in range(5):
                try:
                    output = self._describe_class(item, prompt)
                    logging.info(output)
                    ds_descriptors[item].append(output.description)
                # ollama server failure
                except ollama.ResponseError as e:
                    logging.error(f"Warning! Server error {e}")
                    continue

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(ds_descriptors, f, ensure_ascii=False, indent=4)

        return save_path
