"""
Single-image inference script.

Given `--weights` and an `--image` path, this script:
- loads the trained checkpoint
- applies the same ImageNet normalization as training
- prints top-1 predicted class and probability
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from vision_utils import idx_to_class, load_checkpoint


def make_transforms() -> transforms.Compose:
    # Match evaluation preprocessing.
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def predict_one(model: torch.nn.Module, image_path: Path, device: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = make_transforms()(img).unsqueeze(0).to(device)
    logits = model(x)
    # Return probabilities for all classes.
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict meme template class for an image.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_checkpoint(args.weights, device=device)
    model = bundle.model
    id2class = idx_to_class(bundle.class_to_idx)

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    probs = predict_one(model, image_path=image_path, device=device)
    topk = min(args.topk, probs.numel())
    values, indices = torch.topk(probs, k=topk)

    print("Image:", image_path)
    for score, idx in zip(values.tolist(), indices.tolist(), strict=True):
        print(f"{id2class[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
