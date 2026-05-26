from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import requests


class ComfyUIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/prompt", json=workflow, timeout=30)
        response.raise_for_status()
        return response.json()

    def wait_for_history(self, prompt_id: str, timeout_seconds: int = 600) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            data = response.json()
            if prompt_id in data:
                return data[prompt_id]
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt_id={prompt_id}")

    def download_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        response = requests.get(
            f"{self.base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": folder_type},
            timeout=60,
        )
        response.raise_for_status()
        return response.content


def build_basic_workflow(
    *,
    checkpoint: str,
    positive: str,
    negative: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    filename_prefix: str,
) -> dict[str, Any]:
    return {
        "prompt": {
            "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": Path(checkpoint).name}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
        }
    }


def first_image_ref(history_record: dict[str, Any]) -> dict[str, Any] | None:
    for node_output in history_record.get("outputs", {}).values():
        images = node_output.get("images", [])
        if images:
            return images[0]
    return None
