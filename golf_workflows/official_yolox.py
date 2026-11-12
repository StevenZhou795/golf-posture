"""Native inference for GolfPose's released one- and two-class YOLOX-s detectors.

The checkpoint was trained with MMDetection, but this adapter uses only PyTorch,
OpenCV and torchvision NMS on Windows. Two-class labels are ``person`` and
``club``; the one-class checkpoint labels its single target ``golfer``.
"""

from __future__ import annotations

import builtins
import zipfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from numpy._core import multiarray
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms

from golf_workflows.vitpose_tracking import ROOT, _HistoryBuffer

CHECKPOINT_CANDIDATES = (
    ROOT / "golfpose_checkpoints" / "golfpose_detector_2cls_yolox_s.pth",
    ROOT / "golfpose_detector_2cls_yolox_s.pth",
)
CLASSES = ("person", "club")
GOLFER_CHECKPOINT = ROOT / "golfpose_checkpoints" / "golfpose_detector_1cls_yolox_s.pth"
INPUT_SIZE = 640
STRIDES = (8, 16, 32)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
                                    x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))


class DarknetBottleneck(nn.Module):
    def __init__(self, channels, add_identity=True):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 1)
        self.conv2 = ConvBNAct(channels, channels, 3, padding=1)
        self.add_identity = add_identity

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, add_identity=True):
        super().__init__()
        middle = out_channels // 2
        self.main_conv = ConvBNAct(in_channels, middle, 1)
        self.short_conv = ConvBNAct(in_channels, middle, 1)
        self.final_conv = ConvBNAct(middle * 2, out_channels, 1)
        self.blocks = nn.Sequential(*[DarknetBottleneck(middle, add_identity) for _ in range(blocks)])

    def forward(self, x):
        return self.final_conv(torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        middle = channels // 2
        self.conv1 = ConvBNAct(channels, middle, 1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in (5, 9, 13)])
        self.conv2 = ConvBNAct(middle * 4, channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], dim=1))


class CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Focus(3, 32)
        self.stage1 = nn.Sequential(ConvBNAct(32, 64, 3, 2, 1), CSPLayer(64, 64, 1))
        self.stage2 = nn.Sequential(ConvBNAct(64, 128, 3, 2, 1), CSPLayer(128, 128, 3))
        self.stage3 = nn.Sequential(ConvBNAct(128, 256, 3, 2, 1), CSPLayer(256, 256, 3))
        self.stage4 = nn.Sequential(ConvBNAct(256, 512, 3, 2, 1), SPPBottleneck(512),
                                    CSPLayer(512, 512, 1, add_identity=False))

    def forward(self, x):
        x = self.stage1(self.stem(x))
        stage2 = self.stage2(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stage2, stage3, stage4


class YOLOXPAFPN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (128, 256, 512)
        self.reduce_layers = nn.ModuleList([ConvBNAct(512, 256, 1), ConvBNAct(256, 128, 1)])
        self.top_down_blocks = nn.ModuleList([CSPLayer(512, 256, 1, False), CSPLayer(256, 128, 1, False)])
        self.downsamples = nn.ModuleList([ConvBNAct(128, 128, 3, 2, 1), ConvBNAct(256, 256, 3, 2, 1)])
        self.bottom_up_blocks = nn.ModuleList([CSPLayer(256, 256, 1, False), CSPLayer(512, 512, 1, False)])
        self.out_convs = nn.ModuleList([ConvBNAct(c, 128, 1) for c in channels])

    def forward(self, inputs):
        inner = [inputs[-1]]
        for index in range(2, 0, -1):
            high = self.reduce_layers[2 - index](inner[0])
            inner[0] = high
            merged = torch.cat((F.interpolate(high, scale_factor=2, mode="nearest"), inputs[index - 1]), dim=1)
            inner.insert(0, self.top_down_blocks[2 - index](merged))
        outputs = [inner[0]]
        for index in range(2):
            merged = torch.cat((self.downsamples[index](outputs[-1]), inner[index + 1]), dim=1)
            outputs.append(self.bottom_up_blocks[index](merged))
        return tuple(layer(value) for layer, value in zip(self.out_convs, outputs))


class YOLOXHead(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        def tower():
            return nn.Sequential(ConvBNAct(128, 128, 3, padding=1), ConvBNAct(128, 128, 3, padding=1))
        self.multi_level_cls_convs = nn.ModuleList([tower() for _ in STRIDES])
        self.multi_level_reg_convs = nn.ModuleList([tower() for _ in STRIDES])
        self.multi_level_conv_cls = nn.ModuleList([nn.Conv2d(128, num_classes, 1) for _ in STRIDES])
        self.multi_level_conv_reg = nn.ModuleList([nn.Conv2d(128, 4, 1) for _ in STRIDES])
        self.multi_level_conv_obj = nn.ModuleList([nn.Conv2d(128, 1, 1) for _ in STRIDES])

    def forward(self, features):
        outputs = []
        for i, feature in enumerate(features):
            cls_feature = self.multi_level_cls_convs[i](feature)
            reg_feature = self.multi_level_reg_convs[i](feature)
            outputs.append((self.multi_level_conv_cls[i](cls_feature),
                            self.multi_level_conv_reg[i](reg_feature),
                            self.multi_level_conv_obj[i](reg_feature)))
        return outputs


class GolfPoseYOLOX(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck = YOLOXPAFPN()
        self.bbox_head = YOLOXHead(num_classes)

    def forward(self, x):
        return self.bbox_head(self.neck(self.backbone(x)))


def _load_weights(path: Path, num_classes=2) -> GolfPoseYOLOX:
    allowed = [builtins.getattr, np.ndarray, np.dtype,
               (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               (multiarray.scalar, "numpy.core.multiarray.scalar")]
    allowed += [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "uint8", "bool")]
    allowed.append((_HistoryBuffer, "mmengine.logging.history_buffer.HistoryBuffer"))
    with torch.serialization.safe_globals(allowed):
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=zipfile.is_zipfile(path))
    classes = data.get("meta", {}).get("dataset_meta", {}).get("classes")
    expected = ("golfer",) if num_classes == 1 else ("person", "club")
    actual = (classes,) if isinstance(classes, str) else tuple(classes or ())
    if classes is not None and actual != expected:
        raise ValueError(f"Unexpected detector classes in {path}: {classes}")
    model = GolfPoseYOLOX(num_classes)
    model.load_state_dict(data["state_dict"], strict=True)
    return model.eval()


def _prepare(frame: np.ndarray) -> tuple[torch.Tensor, float]:
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float(), scale


def _decode(outputs, score_threshold: float, iou_threshold: float, scale: float):
    boxes, scores, labels = [], [], []
    for stride, (cls_score, bbox_pred, objectness) in zip(STRIDES, outputs):
        _, _, height, width = cls_score.shape
        yy, xx = torch.meshgrid(torch.arange(height, device=bbox_pred.device),
                               torch.arange(width, device=bbox_pred.device), indexing="ij")
        prior = torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(bbox_pred) * stride
        pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        center = pred[:, :2] * stride + prior
        size = pred[:, 2:].exp() * stride
        decoded = torch.cat((center - size / 2, center + size / 2), dim=1)
        num_classes = cls_score.shape[1]
        class_scores, class_labels = cls_score.sigmoid().permute(0, 2, 3, 1).reshape(-1, num_classes).max(dim=1)
        combined = class_scores * objectness.sigmoid().reshape(-1)
        keep = combined >= score_threshold
        boxes.append(decoded[keep])
        scores.append(combined[keep])
        labels.append(class_labels[keep])
    if not boxes or not any(len(item) for item in boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)
    boxes = torch.cat(boxes)
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    kept = []
    for label in labels.unique().tolist():
        indexes = torch.where(labels == label)[0]
        if len(indexes):
            kept.append(indexes[nms(boxes[indexes], scores[indexes], iou_threshold)])
    keep = torch.cat(kept) if kept else torch.empty(0, dtype=torch.long)
    keep = keep[scores[keep].argsort(descending=True)]
    return (boxes[keep].div(scale).cpu().numpy(), scores[keep].cpu().numpy(),
            labels[keep].cpu().numpy())


class OfficialClubDetector:
    """Released two-class YOLOX-s detector with a club-box convenience API."""

    def __init__(self, checkpoint: Path | None = None, score_threshold: float = 0.01):
        checkpoint = Path(checkpoint) if checkpoint else next((p for p in CHECKPOINT_CANDIDATES if p.is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError("golfpose_detector_2cls_yolox_s.pth was not found")
        self.checkpoint = checkpoint
        self.score_threshold = score_threshold
        self.model = _load_weights(checkpoint)
        self.metadata = {
            "club_crop_backend": "official-yolox-2cls",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "classes": list(CLASSES),
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "score_threshold": score_threshold,
            "nms_iou_threshold": 0.65,
            "strict_parameter_load": True,
        }

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> dict:
        tensor, scale = _prepare(frame)
        tensor = tensor.to(next(self.model.parameters()).device)
        started = perf_counter()
        boxes, scores, labels = _decode(self.model(tensor), self.score_threshold, 0.65, scale)
        height, width = frame.shape[:2]
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
        return {"boxes": boxes, "scores": scores, "labels": labels,
                "inference_seconds": round(perf_counter() - started, 3)}

    def club_box(self, frame: np.ndarray, result=None, person_box=None, anchor=None) -> tuple[np.ndarray | None, dict]:
        result = self.predict(frame) if result is None else result
        indexes = np.where(result["labels"] == 1)[0]
        candidate_count = len(indexes)
        if person_box is not None and np.isfinite(person_box).all() and len(indexes):
            candidates = result["boxes"][indexes]
            height = max(float(person_box[3]-person_box[1]), 1.)
            gap = np.maximum(0, np.maximum(person_box[:2]-candidates[:, 2:], candidates[:, :2]-person_box[2:]))
            associated = np.linalg.norm(gap, axis=1) <= .75*height
            if anchor is not None and np.isfinite(anchor).all():
                hand_gap = np.maximum(0, np.maximum(candidates[:, :2]-anchor, anchor-candidates[:, 2:]))
                associated &= np.linalg.norm(hand_gap, axis=1) <= .35*height
            indexes = indexes[associated]
        chosen = int(indexes[np.argmax(result["scores"][indexes])]) if len(indexes) else None
        box = None if chosen is None else result["boxes"][chosen].astype(np.float64)
        details = {
            "detector": self.metadata,
            "club_candidates": int(candidate_count),
            "associated_club_candidates": int(len(indexes)),
            "selected_club_score": None if chosen is None else round(float(result["scores"][chosen]), 6),
            "selected_club_bbox_xyxy": None if box is None else box.tolist(),
            "inference_seconds": result["inference_seconds"],
        }
        return box, details


def select_box(result, label=0, minimum_score=.25, previous=None, require_overlap=False):
    """Choose by continuity when anchored, otherwise by detection confidence.

    With `require_overlap`, reject incompatible displacement/scale as well as
    missing overlap instead of jumping to a higher-confidence bystander.
    """
    boxes = np.asarray(result["boxes"], dtype=float).reshape(-1, 4)
    scores = np.asarray(result["scores"], dtype=float)
    indexes = np.flatnonzero((np.asarray(result["labels"]) == label) & np.isfinite(scores)
        & (scores >= minimum_score) & np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    if not len(indexes):
        return None, None
    if previous is not None and np.isfinite(previous).all():
        candidates = boxes[indexes]
        wh = np.maximum(0, np.minimum(candidates[:, 2:], previous[2:])
                        - np.maximum(candidates[:, :2], previous[:2]))
        intersection = wh.prod(axis=1)
        union = (candidates[:, 2:] - candidates[:, :2]).prod(axis=1) + np.prod(previous[2:] - previous[:2]) - intersection
        overlaps = intersection / np.maximum(union, 1e-6)
        size = candidates[:, 2:] - candidates[:, :2]
        reference_size = np.maximum(np.asarray(previous[2:]) - previous[:2], 1.)
        displacement = np.linalg.norm(
            ((candidates[:, :2] + candidates[:, 2:] - previous[:2] - previous[2:]) / 2)
            / reference_size, axis=1)
        scale = size / reference_size
        compatible = (overlaps >= .1)
        if require_overlap:
            compatible &= (displacement <= .6) & (scale.min(axis=1) >= .45) & (scale.max(axis=1) <= 2.)
        matching = indexes[compatible]
        if len(matching):
            # Confidence breaks ties; it must not override identity continuity.
            affinity = overlaps - .35 * displacement + .05 * scores[indexes]
            chosen = int(indexes[np.where(compatible, affinity, -np.inf).argmax()])
            return boxes[chosen].copy(), float(scores[chosen])
        elif require_overlap:
            return None, None
    chosen = int(indexes[np.argmax(scores[indexes])])
    return boxes[chosen].copy(), float(scores[chosen])


class OfficialGolferDetector(OfficialClubDetector):
    """One-class golfer detector used to crop the joint 22-landmark pose input."""

    def __init__(self, checkpoint=GOLFER_CHECKPOINT, score_threshold=.25):
        self.checkpoint = Path(checkpoint)
        self.score_threshold = score_threshold
        self.model = _load_weights(self.checkpoint, num_classes=1)
        self.metadata = dict(detector_backend="official-yolox-1cls",
            checkpoint=str(self.checkpoint.relative_to(ROOT)), classes=["golfer"],
            input_size=[INPUT_SIZE, INPUT_SIZE], score_threshold=score_threshold,
            nms_iou_threshold=.65, strict_parameter_load=True)

    def golfer_box(self, frame, previous=None):
        result = self.predict(frame)
        box, score = select_box(result, minimum_score=self.score_threshold, previous=previous)
        return box, dict(detector=self.metadata, selected_golfer_bbox_xyxy=None if box is None else box.tolist(),
            selected_golfer_score=score, candidates=len(result["boxes"]),
            inference_seconds=result["inference_seconds"])


def load_club_crop_backend(name: str):
    if name == "segmentation":
        return None
    if name == "official-yolox-2cls":
        return OfficialClubDetector()
    raise ValueError(f"Unknown club crop backend: {name}")
