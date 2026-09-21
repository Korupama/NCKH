#!/usr/bin/env python3
"""Run SST soccer-object detection on one image frame.

This loader is designed for the public checkpoint from rvandeghen/SST.  The
checkpoint was produced with an older torchvision Faster R-CNN implementation,
so the code includes small state-dict migrations for modern torchvision.

Security note: the original checkpoint contains a pickled AnchorGenerator
object.  Therefore torch.load(..., weights_only=False) is required.  Only load
checkpoints from sources you trust.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import pil_to_tensor


CLASS_NAMES: Dict[int, str] = {
    1: "Ball",
    2: "Player",
    3: "Goalkeeper",
    4: "Main referee",
    5: "Side referee",
    6: "Staff members",
}

# The colors are intentionally high-contrast on a green soccer pitch.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 215, 0),
    2: (0, 200, 255),
    3: (255, 80, 80),
    4: (255, 0, 255),
    5: (180, 80, 255),
    6: (255, 150, 0),
}

# Fallback values copied from the SST training script.  Normally the values are
# read from checkpoint["anchor_generator"].
DEFAULT_ANCHOR_SIZES: Tuple[Tuple[float, ...], ...] = tuple(
    (x * 0.337, x * 0.517, x * 1.939) for x in (32, 64, 128, 256, 512)
)
DEFAULT_ASPECT_RATIOS: Tuple[Tuple[float, ...], ...] = (
    (0.289, 1.0, 3.458),
) * len(DEFAULT_ANCHOR_SIZES)


class SSTCheckpointError(RuntimeError):
    """Raised when an SST checkpoint cannot be reconstructed safely."""


def _install_legacy_anchor_pickle_shim() -> None:
    """Provide models.anchor_utils.AnchorGenerator for unpickling old SST files.

    The original checkpoint stores the whole custom AnchorGenerator object, not
    only tensors.  A minimal nn.Module-compatible class is enough to recover its
    attributes (sizes and aspect_ratios); inference then uses torchvision's
    current AnchorGenerator implementation.
    """

    existing = sys.modules.get("models.anchor_utils")
    if existing is not None and hasattr(existing, "AnchorGenerator"):
        return

    # Remove a partially imported, incompatible legacy package if one exists.
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]

    models_package = types.ModuleType("models")
    models_package.__path__ = []  # Mark it as a package.

    anchor_module = types.ModuleType("models.anchor_utils")

    class AnchorGeneratorShim(torch.nn.Module):
        def __init__(
            self,
            sizes: Sequence[Sequence[float]] = DEFAULT_ANCHOR_SIZES,
            aspect_ratios: Sequence[Sequence[float]] = DEFAULT_ASPECT_RATIOS,
        ) -> None:
            super().__init__()
            self.sizes = tuple(tuple(float(v) for v in group) for group in sizes)
            self.aspect_ratios = tuple(
                tuple(float(v) for v in group) for group in aspect_ratios
            )
            self.cell_anchors: List[torch.Tensor] = []

    # Pickle resolves classes by module + name.
    AnchorGeneratorShim.__name__ = "AnchorGenerator"
    AnchorGeneratorShim.__qualname__ = "AnchorGenerator"
    AnchorGeneratorShim.__module__ = "models.anchor_utils"

    anchor_module.AnchorGenerator = AnchorGeneratorShim
    models_package.anchor_utils = anchor_module
    sys.modules["models"] = models_package
    sys.modules["models.anchor_utils"] = anchor_module


def _torch_load_trusted_checkpoint(path: Path, device: torch.device) -> Any:
    _install_legacy_anchor_pickle_shim()
    kwargs: Dict[str, Any] = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # Required because SST saves a custom AnchorGenerator object.
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _extract_state_dict(checkpoint: Any) -> MutableMapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    if not isinstance(state, Mapping) or not state:
        raise SSTCheckpointError("Checkpoint does not contain a non-empty model state_dict.")

    result: MutableMapping[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        result[clean_key] = value

    if not result:
        raise SSTCheckpointError("No tensor parameters were found in the checkpoint.")
    return result


def _infer_num_classes(state_dict: Mapping[str, torch.Tensor]) -> int:
    candidate_keys = (
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
    )
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim >= 1:
            return int(tensor.shape[0])
    # SST uses 6 foreground categories plus background.
    return 7


def _normalise_nested_floats(
    value: Any,
    fallback: Tuple[Tuple[float, ...], ...],
) -> Tuple[Tuple[float, ...], ...]:
    try:
        normalised = tuple(tuple(float(v) for v in group) for group in value)
        if normalised and all(group for group in normalised):
            return normalised
    except (TypeError, ValueError):
        pass
    return fallback


def _extract_anchor_configuration(
    checkpoint: Any,
) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...]]:
    legacy_anchor = checkpoint.get("anchor_generator") if isinstance(checkpoint, Mapping) else None
    sizes = _normalise_nested_floats(
        getattr(legacy_anchor, "sizes", None), DEFAULT_ANCHOR_SIZES
    )
    aspect_ratios = _normalise_nested_floats(
        getattr(legacy_anchor, "aspect_ratios", None), DEFAULT_ASPECT_RATIOS
    )
    if len(sizes) != len(aspect_ratios):
        raise SSTCheckpointError(
            "Anchor sizes and aspect ratios have different numbers of feature levels."
        )
    return sizes, aspect_ratios


def _state_key_candidates(old_key: str) -> Iterable[str]:
    """Yield plausible modern torchvision names for one legacy parameter."""
    yield old_key

    # torchvision changed RPNHead.conv from Conv2d to nested Sequential blocks.
    rpn_match = re.fullmatch(r"rpn\.head\.conv\.(.+)", old_key)
    if rpn_match:
        suffix = rpn_match.group(1)
        for zero_depth in (1, 2, 3):
            yield "rpn.head.conv." + ("0." * zero_depth) + suffix

    # FeaturePyramidNetwork similarly gained one or more wrapper levels.
    fpn_match = re.fullmatch(
        r"(backbone\.fpn\.(?:inner_blocks|layer_blocks)\.\d+)\.(.+)", old_key
    )
    if fpn_match:
        prefix, suffix = fpn_match.groups()
        for zero_depth in (1, 2, 3):
            yield prefix + "." + ("0." * zero_depth) + suffix


def _migrate_state_dict(
    legacy_state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> MutableMapping[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    migrated: MutableMapping[str, torch.Tensor] = {}

    for old_key, value in legacy_state.items():
        target_key = old_key
        for candidate in _state_key_candidates(old_key):
            if candidate in model_keys:
                target_key = candidate
                break
        migrated[target_key] = value

    return migrated


def build_sst_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a public SST checkpoint into a modern torchvision Faster R-CNN."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(device)
    checkpoint = _torch_load_trusted_checkpoint(checkpoint_path, resolved_device)
    legacy_state = _extract_state_dict(checkpoint)
    num_classes = _infer_num_classes(legacy_state)
    anchor_sizes, aspect_ratios = _extract_anchor_configuration(checkpoint)

    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=5,
    )
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
    )

    migrated_state = _migrate_state_dict(legacy_state, model)
    incompatible = model.load_state_dict(migrated_state, strict=False)

    # BatchNorm checkpoints may contain this harmless bookkeeping buffer, while
    # FrozenBatchNorm2d does not.  All other mismatches are treated as errors.
    missing = list(incompatible.missing_keys)
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.endswith("num_batches_tracked")
    ]
    if missing or unexpected:
        details = [
            "The SST checkpoint does not fully match the reconstructed model.",
            f"Missing keys ({len(missing)}): {missing[:20]}",
            f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}",
            "Use the original repo environment if this is a differently trained checkpoint.",
        ]
        raise SSTCheckpointError("\n".join(details))

    model.to(resolved_device)
    model.eval()

    metadata: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "num_classes_including_background": num_classes,
        "anchor_sizes": anchor_sizes,
        "aspect_ratios": aspect_ratios,
    }
    if isinstance(checkpoint, Mapping):
        if "map" in checkpoint:
            try:
                metadata["checkpoint_map"] = float(checkpoint["map"])
            except (TypeError, ValueError):
                metadata["checkpoint_map"] = checkpoint["map"]
        if "epoch" in checkpoint:
            metadata["epoch"] = checkpoint["epoch"]

    return model, metadata


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


@torch.inference_mode()
def predict_frame(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Return raw Faster R-CNN output for one RGB PIL image."""

    rgb_image = image.convert("RGB")
    tensor = pil_to_tensor(rgb_image).float().div(255.0).to(device)
    output = model([tensor])[0]
    return {key: value.detach().cpu() for key, value in output.items()}


def _load_font(font_size: int) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "arialbd.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_predictions(
    image: Image.Image,
    prediction: Mapping[str, torch.Tensor],
    score_threshold: float = 0.50,
    ball_threshold: float = 0.20,
    max_detections: int = 100,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Draw boxes and return both the annotated image and structured detections."""

    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be in [0, 1].")
    if not (0.0 <= ball_threshold <= 1.0):
        raise ValueError("ball_threshold must be in [0, 1].")

    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    line_width = max(2, round(min(width, height) / 300))
    font_size = max(12, round(min(width, height) / 42))
    font = _load_font(font_size)

    boxes = prediction.get("boxes", torch.empty((0, 4)))
    labels = prediction.get("labels", torch.empty((0,), dtype=torch.long))
    scores = prediction.get("scores", torch.empty((0,)))

    detections: List[Dict[str, Any]] = []
    for box, label_tensor, score_tensor in zip(boxes, labels, scores):
        label_id = int(label_tensor.item())
        score = float(score_tensor.item())
        threshold = ball_threshold if label_id == 1 else score_threshold
        if score < threshold:
            continue
        if len(detections) >= max_detections:
            break

        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        x1 = min(max(x1, 0.0), width - 1.0)
        y1 = min(max(y1, 0.0), height - 1.0)
        x2 = min(max(x2, x1 + 1.0), float(width))
        y2 = min(max(y2, y1 + 1.0), float(height))

        class_name = CLASS_NAMES.get(label_id, f"Class {label_id}")
        color = CLASS_COLORS.get(label_id, (255, 255, 255))
        caption = f"{class_name} {score:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        # Make the tiny ball detection easier to see in a full-resolution frame.
        if label_id == 1:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            radius = max(7, line_width * 3)
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline=color,
                width=line_width,
            )
            draw.line((cx - radius, cy, cx + radius, cy), fill=color, width=line_width)
            draw.line((cx, cy - radius, cx, cy + radius), fill=color, width=line_width)

        text_bbox = draw.textbbox((0, 0), caption, font=font, stroke_width=0)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        pad = max(2, line_width)
        text_x = x1
        text_y = max(0.0, y1 - text_height - 2 * pad)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 2 * pad, text_y + text_height + 2 * pad),
            fill=color,
        )
        draw.text(
            (text_x + pad, text_y + pad),
            caption,
            fill=(0, 0, 0),
            font=font,
        )

        detections.append(
            {
                "label_id": label_id,
                "label": class_name,
                "score": round(score, 6),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    return result, detections


def process_frame(
    checkpoint_path: str | Path,
    input_image_path: str | Path,
    output_image_path: str | Path,
    *,
    device: str = "auto",
    score_threshold: float = 0.50,
    ball_threshold: float = 0.20,
    max_detections: int = 100,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    """High-level API: load model, process one frame, and save the image."""

    resolved_device = resolve_device(device)
    model, metadata = build_sst_model(checkpoint_path, resolved_device)

    input_path = Path(input_image_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    output_path = Path(output_image_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as loaded:
        image = loaded.convert("RGB")
    prediction = predict_frame(model, image, resolved_device)
    annotated, detections = annotate_predictions(
        image,
        prediction,
        score_threshold=score_threshold,
        ball_threshold=ball_threshold,
        max_detections=max_detections,
    )
    annotated.save(output_path)
    metadata["device"] = str(resolved_device)
    metadata["input_image"] = str(input_path)
    metadata["output_image"] = str(output_path)
    metadata["num_detections_after_threshold"] = len(detections)
    return output_path, detections, metadata


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect soccer ball/people in one frame using the SST checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to SST model.pth")
    parser.add_argument("--input", required=True, help="Path to the input frame image")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path; default: <input_stem>_sst.<input_ext>",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON path for boxes, labels, scores, and metadata",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--ball-threshold", type=float, default=0.20)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_sst{input_path.suffix or '.jpg'}")
    )

    saved_path, detections, metadata = process_frame(
        checkpoint_path=args.checkpoint,
        input_image_path=input_path,
        output_image_path=output_path,
        device=args.device,
        score_threshold=args.score_threshold,
        ball_threshold=args.ball_threshold,
        max_detections=args.max_detections,
    )

    print(f"Saved processed image: {saved_path}")
    print(f"Detections kept: {len(detections)}")
    for detection in detections:
        print(
            f"- {detection['label']}: {detection['score']:.3f}, "
            f"box={detection['bbox_xyxy']}"
        )

    if args.json_output:
        json_path = Path(args.json_output).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"metadata": metadata, "detections": detections},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
