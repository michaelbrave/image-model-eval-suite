from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math

from PIL import Image, ImageStat


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


class BrightnessContrastScorer(Scorer):
    name = "brightness-contrast"
    score_kind = "technical_image_stats"

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        image = Image.open(image_path).convert("RGB")
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0]) / 255.0
        contrast = float(stat.stddev[0]) / 128.0
        contrast = max(0.0, min(1.0, contrast))
        # Objective technical score only; not a claim of aesthetic quality.
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


class ImprovedAestheticPredictorScorer(Scorer):
    name = "improved-aesthetic-predictor"
    score_kind = "aesthetic"

    def __init__(self, weight_path: str | None = None) -> None:
        self.weight_path = Path(weight_path) if weight_path else None

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        if self.weight_path is None or not self.weight_path.exists():
            raise RuntimeError(
                "improved-aesthetic-predictor requires --weight-path pointing to real predictor weights"
            )
        try:
            import clip  # type: ignore
            import numpy as np  # type: ignore
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install torch, clip, and numpy before using improved-aesthetic-predictor") from exc

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = MLP(768)
        predictor.load_state_dict(torch.load(self.weight_path, map_location=torch.device("cpu")))
        predictor.to(device)
        predictor.eval()
        clip_model, preprocess = clip.load("ViT-L/14", device=device)

        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_array = image_features.cpu().detach().numpy()
            norm = np.linalg.norm(image_array, ord=2, axis=-1, keepdims=True)
            norm[norm == 0] = 1
            normalized = image_array / norm
            prediction = predictor(torch.from_numpy(normalized).to(device).float())
        raw = float(prediction.item())
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=raw,
            normalized_score=None,
            metadata={"weight_path": str(self.weight_path), "style_hint": style_hint},
        )


class ImageRewardScorer(Scorer):
    name = "image-reward"
    score_kind = "prompt_reward"

    def score(self, image_path: Path, prompt: str | None = None, style_hint: str | None = None) -> ScoreResult:
        if not prompt:
            raise ValueError("image-reward requires the rendered prompt")
        try:
            import ImageReward as RM  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install ImageReward before using image-reward") from exc
        model = RM.load("ImageReward-v1.0")
        raw = float(model.score(prompt, str(image_path)))
        return ScoreResult(
            scorer_name=self.name,
            score_kind=self.score_kind,
            raw_score=raw,
            normalized_score=None,
            metadata={"style_hint": style_hint},
        )


SCORERS = {
    "brightness-contrast": BrightnessContrastScorer,
    "improved-aesthetic-predictor": ImprovedAestheticPredictorScorer,
    "image-reward": ImageRewardScorer,
}


def make_scorer(name: str, **kwargs: Any) -> Scorer:
    if name not in SCORERS:
        raise KeyError(f"Unknown scorer: {name}")
    cls = SCORERS[name]
    if name == "improved-aesthetic-predictor":
        return cls(weight_path=kwargs.get("weight_path"))  # type: ignore[call-arg]
    return cls()  # type: ignore[call-arg]


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_run(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
