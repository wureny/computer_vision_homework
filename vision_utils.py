from __future__ import annotations

"""
Model utilities for the meme template classification project.

- `build_mobilenetv2`: create MobileNetV2 and replace the classifier head
- `save_checkpoint` / `load_checkpoint`: store and restore model weights + label mapping
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision import models


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    class_to_idx: dict[str, int]


def build_mobilenetv2(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    # When `pretrained=True`, TorchVision may download ImageNet weights on first use.
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    class_to_idx: dict[str, int],
    meta: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "meta": meta or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str) -> ModelBundle:
    ckpt = torch.load(path, map_location=device)
    if "state_dict" not in ckpt or "class_to_idx" not in ckpt:
        raise ValueError("Invalid checkpoint: missing 'state_dict' or 'class_to_idx'.")
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)
    model = build_mobilenetv2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return ModelBundle(model=model, class_to_idx=class_to_idx)


def save_label_map(path: str | Path, class_to_idx: dict[str, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")


def idx_to_class(class_to_idx: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in class_to_idx.items()}
