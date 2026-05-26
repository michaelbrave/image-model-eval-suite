from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import math

from PIL import Image, ImageStat

REFERENCE_ROOT = Path("/home/mike/image-workflow/references/scorers/external")
IMPROVED_WEIGHT = REFERENCE_ROOT / "improved-aesthetic-predictor" / "sac+logos+ava1-l14-linearMSE.pth"
SD_CHAD_WEIGHT = REFERENCE_ROOT / "SD-Chad" / "chadscorer.pth"
AESTHETICS_SCORER_ROOT = REFERENCE_ROOT / "aesthetics-scorer"
AESTHETICS_SCORER_WEIGHT = AESTHETICS_SCORER_ROOT / "aesthetics_scorer" / "models" / "aesthetics_scorer_rating_openclip_vit_h_14.pth"
AESTHETICS_ARTIFACT_WEIGHT = AESTHETICS_SCORER_ROOT / "aesthetics_scorer" / "models" / "aesthetics_scorer_artifacts_openclip_vit_h_14.pth"


@dataclass(frozen=True)
class ScoreResult:
    scorer_name: str
    score_kind: str
    raw_score: float
    normalized_score: float | None
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class Scorer:
    name: str
    score_kind: str

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        raise NotImplementedError


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_1_to_10(value: float) -> float:
    return clamp01((value - 1.0) / 9.0)


class BrightnessContrastScorer(Scorer):
    name = "brightness-contrast"
    score_kind = "technical_image_stats"

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        image = Image.open(image_path).convert("RGB")
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0]) / 255.0
        contrast = float(stat.stddev[0]) / 128.0
        contrast = clamp01(contrast)
        exposure_balance = 1.0 - min(abs(brightness - 0.5) * 2.0, 1.0)
        technical_score = (0.55 * exposure_balance) + (0.45 * contrast)
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=technical_score,
            normalized_score=technical_score,
            metadata={
                "brightness_mean_0_1": brightness,
                "contrast_stddev_0_1": contrast,
                "width": image.width,
                "height": image.height,
            },
        )


class ClipMlpAestheticScorer(Scorer):
    name = "clip-mlp-aesthetic"
    score_kind = "aesthetic"
    weight_path: Path

    def __init__(self, weight_path: str | Path | None = None) -> None:
        self.weight_path = Path(weight_path) if weight_path else self.weight_path

    @staticmethod
    @lru_cache(maxsize=2)
    def _runtime(weight_path: str) -> dict[str, Any]:
        try:
            import clip  # type: ignore
            import numpy as np  # type: ignore
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install torch, openai-clip, and numpy before using CLIP MLP aesthetic scorers") from exc

        class MLP(nn.Module):
            def __init__(self, input_size: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.layers(x)

        path = Path(weight_path)
        if not path.exists():
            raise RuntimeError(f"Missing scorer weight: {path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = MLP(768)
        predictor.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        predictor.to(device)
        predictor.eval()
        clip_model, preprocess = clip.load("ViT-L/14", device=device)
        return {"torch": torch, "np": np, "predictor": predictor, "clip_model": clip_model, "preprocess": preprocess, "device": device}

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        runtime = self._runtime(str(self.weight_path))
        torch = runtime["torch"]
        np = runtime["np"]
        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = runtime["preprocess"](pil_image).unsqueeze(0).to(runtime["device"])
        with torch.no_grad():
            image_features = runtime["clip_model"].encode_image(image_tensor)
            image_array = image_features.cpu().detach().numpy()
            norm = np.linalg.norm(image_array, ord=2, axis=-1, keepdims=True)
            norm[norm == 0] = 1
            normalized = image_array / norm
            prediction = runtime["predictor"](torch.from_numpy(normalized).to(runtime["device"]).float())
        raw = float(prediction.item())
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=raw,
            normalized_score=normalize_1_to_10(raw),
            metadata={"weight_path": str(self.weight_path), "style_hint": style_hint},
        )


class ImprovedAestheticPredictorScorer(ClipMlpAestheticScorer):
    name = "improved-aesthetic-predictor"
    weight_path = IMPROVED_WEIGHT


class SdChadScorer(ClipMlpAestheticScorer):
    name = "sd-chad"
    weight_path = SD_CHAD_WEIGHT


class AestheticsScorer(Scorer):
    name = "aesthetics-scorer"
    score_kind = "aesthetic"

    @staticmethod
    @lru_cache(maxsize=1)
    def _runtime() -> dict[str, Any]:
        try:
            import sys
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers before using aesthetics-scorer") from exc
        sys.path.insert(0, str(AESTHETICS_SCORER_ROOT / "aesthetics_scorer"))
        try:
            from model import load_model, preprocess  # type: ignore
        finally:
            sys.path.pop(0)
        if not AESTHETICS_SCORER_WEIGHT.exists():
            raise RuntimeError(f"Missing aesthetics-scorer weight: {AESTHETICS_SCORER_WEIGHT}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        clip_model = CLIPModel.from_pretrained(model_id)
        vision_model = clip_model.vision_model.to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        rating_model = load_model(str(AESTHETICS_SCORER_WEIGHT), device=device).to(device).eval()
        artifact_model = None
        if AESTHETICS_ARTIFACT_WEIGHT.exists():
            artifact_model = load_model(str(AESTHETICS_ARTIFACT_WEIGHT), device=device).to(device).eval()
        return {"torch": torch, "preprocess": preprocess, "vision_model": vision_model, "processor": processor, "rating_model": rating_model, "artifact_model": artifact_model, "device": device, "model_id": model_id}

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        runtime = self._runtime()
        torch = runtime["torch"]
        image = Image.open(image_path).convert("RGB")
        inputs = runtime["processor"](images=image, return_tensors="pt").to(runtime["device"])
        with torch.no_grad():
            vision_output = runtime["vision_model"](**inputs)
            embedding = runtime["preprocess"](vision_output.pooler_output)
            rating = runtime["rating_model"](embedding)
            artifact = runtime["artifact_model"](embedding) if runtime["artifact_model"] is not None else None
        raw = float(rating.detach().cpu().item())
        artifact_raw = float(artifact.detach().cpu().item()) if artifact is not None else None
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=raw,
            normalized_score=normalize_1_to_10(raw),
            metadata={"model_id": runtime["model_id"], "artifact_score_raw": artifact_raw, "style_hint": style_hint},
        )


class CafeAestheticScorer(Scorer):
    name = "cafe-aesthetic"
    score_kind = "aesthetic_classifier"

    @staticmethod
    @lru_cache(maxsize=1)
    def _runtime() -> dict[str, Any]:
        try:
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers before using cafe-aesthetic") from exc
        device = 0 if torch.cuda.is_available() else -1
        return {
            "aesthetic": pipeline("image-classification", "cafeai/cafe_aesthetic", device=device),
            "style": pipeline("image-classification", "cafeai/cafe_style", device=device),
            "waifu": pipeline("image-classification", "cafeai/cafe_waifu", device=device),
            "device": device,
        }

    @staticmethod
    def _positive_score(labels: list[dict[str, Any]]) -> float:
        best_positive = 0.0
        for item in labels:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if "not" in label or "low" in label or "worst" in label or "bad" in label:
                continue
            best_positive = max(best_positive, score)
        return best_positive if best_positive > 0 else float(labels[0].get("score", 0.0))

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        runtime = self._runtime()
        image = Image.open(image_path).convert("RGB")
        aesthetic = runtime["aesthetic"](image, top_k=5)
        style = runtime["style"](image, top_k=5)
        waifu = runtime["waifu"](image, top_k=5)
        raw = self._positive_score(aesthetic)
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=raw,
            normalized_score=clamp01(raw),
            metadata={"aesthetic": aesthetic, "style": style, "waifu": waifu, "style_hint": style_hint},
        )


class ImageRewardScorer(Scorer):
    name = "image-reward"
    score_kind = "prompt_reward"

    @staticmethod
    @lru_cache(maxsize=1)
    def _runtime() -> Any:
        try:
            import ImageReward as RM  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install ImageReward before using image-reward") from exc
        return RM.load("ImageReward-v1.0")

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        if not prompt:
            raise ValueError("image-reward requires the rendered prompt")
        model = self._runtime()
        raw = float(model.score(prompt, str(image_path)))
        normalized = 1.0 / (1.0 + math.exp(-raw))
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=raw,
            normalized_score=normalized,
            metadata={"style_hint": style_hint, "normalization": "sigmoid(raw_score)"},
        )


SCORERS = {
    "brightness-contrast": BrightnessContrastScorer,
    "improved-aesthetic-predictor": ImprovedAestheticPredictorScorer,
    "sd-chad": SdChadScorer,
    "aesthetics-scorer": AestheticsScorer,
    "cafe-aesthetic": CafeAestheticScorer,
    "image-reward": ImageRewardScorer,
}


def make_scorer(name: str, **kwargs: Any) -> Scorer:
    if name not in SCORERS:
        raise KeyError(f"Unknown scorer: {name}")
    cls = SCORERS[name]
    if name == "improved-aesthetic-predictor" and kwargs.get("weight_path"):
        return cls(weight_path=kwargs.get("weight_path"))  # type: ignore[call-arg]
    return cls()  # type: ignore[call-arg]


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_run(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
