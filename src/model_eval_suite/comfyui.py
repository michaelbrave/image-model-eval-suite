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


def build_z_image_turbo_workflow(
    *,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    positive: str,
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
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "lumina2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}},
            "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
        }
    }


def build_qwen_image_workflow(
    *,
    unet_name: str,
    clip_name: str,
    vae_name: str,
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
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "qwen_image"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            "6": {"class_type": "EmptyQwenImageLayeredLatentImage", "inputs": {"width": width, "height": height, "layers": 0, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
        }
    }


def build_lumina2_workflow(
    *,
    unet_name: str,
    clip_name: str,
    vae_name: str,
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
    system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts."
    return {
        "prompt": {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "lumina2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 6}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{system_prompt} <Prompt Start> {positive}", "clip": ["2", 0]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0, "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]}},
        }
    }


def build_pixeldit_workflow(
    *,
    unet_name: str,
    clip_name: str,
    vae_name: str,
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
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "pixeldit", "device": "cpu"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]}},
        }
    }


WORKFLOW_BUILDERS: dict[str, Any] = {
    "checkpoint": build_basic_workflow,
    "z-image-turbo": build_z_image_turbo_workflow,
    "pixeldit": build_pixeldit_workflow,
    "qwen-image": build_qwen_image_workflow,
    "lumina2": build_lumina2_workflow,
}


def get_model_params(workflow_type: str) -> dict[str, str]:
    if workflow_type == "checkpoint":
        return {}
    elif workflow_type == "z-image-turbo":
        return {"unet_name": "z_image_turbo_nvfp4.safetensors", "clip_name": "qwen_3_4b_fp8_mixed.safetensors", "vae_name": "ae.safetensors"}
    elif workflow_type == "pixeldit":
        return {"unet_name": "pixeldit_1300m_1024px_bf16.safetensors", "clip_name": "gemma_2_2b_it_elm_fp8_scaled.safetensors", "vae_name": "pixel_space"}
    elif workflow_type == "qwen-image":
        return {"unet_name": "qwen_image_fp8_e4m3fn.safetensors", "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "vae_name": "qwen_image_vae.safetensors"}
    elif workflow_type == "lumina2":
        return {"unet_name": "lumina_2_model_bf16.safetensors", "clip_name": "gemma_2_2b_fp16.safetensors", "vae_name": "ae.safetensors"}
    raise ValueError(f"Unknown workflow_type: {workflow_type}")


def first_image_ref(history_record: dict[str, Any]) -> dict[str, Any] | None:
    for node_output in history_record.get("outputs", {}).values():
        images = node_output.get("images", [])
        if images:
            return images[0]
    return None
