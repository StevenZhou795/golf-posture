"""ViTPose-Huge top-down body pose, compatible with the existing single-golfer workflow.

The inference graph follows MMPreTrain VisionTransformer and MMPose HeatmapHead.
It uses the saved COCO checkpoint, YOLO26 boxes, UDP crops/decoding and flip testing.
No replacement packages or global runtime patches are required.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from third_party.golfpose.vitpose_udp import get_heatmap_maximum, refine_keypoints_dark_udp

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "golfpose_checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
BACKENDS = ("yolo26n-pose", "vitpose-huge")
COCO_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(1280, 3840)
        self.proj = nn.Linear(1280, 1280)

    def forward(self, x):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, 16, 80).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, channels))


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(1280, 5120), nn.GELU()), nn.Linear(5120, 1280))

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(1280, eps=1e-6)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(1280, eps=1e-6)
        self.ffn = FFN()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 193, 1280))
        # The older MMPreTrain export retains this unused parameter as well.
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1280))
        self.patch_embed = nn.Module()
        self.patch_embed.projection = nn.Conv2d(3, 1280, 16, stride=16, padding=2)
        self.layers = nn.ModuleList([Encoder() for _ in range(32)])
        self.ln1 = nn.LayerNorm(1280, eps=1e-6)

    def forward(self, image):
        patches = self.patch_embed.projection(image)
        h, w = patches.shape[-2:]
        if (h, w) != (16, 12):
            raise ValueError(f"Expected 16x12 image patches, got {h}x{w}")
        # The released checkpoint retains a class-position slot, but no class token.
        x = patches.flatten(2).transpose(1, 2) + self.pos_embed[:, 1:]
        for layer in self.layers:
            x = layer(x)
        return self.ln1(x).transpose(1, 2).reshape(image.shape[0], 1280, h, w)


class HeatmapHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.final_layer = nn.Conv2d(256, 17, 1)

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


class ViTPoseHuge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = HeatmapHead()

    def forward(self, image):
        return self.head(self.backbone(image))


def load_checkpoint(path=CHECKPOINT):
    from numpy._core import multiarray
    allowed = [np.ndarray, np.dtype, (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               (multiarray.scalar, "numpy.core.multiarray.scalar")]
    allowed += [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "uint8", "bool")]
    with torch.serialization.safe_globals(allowed):
        data = torch.load(path, map_location="cpu", weights_only=True)
    # Meta construction avoids allocating a second 2.55 GB set of parameters.
    with torch.device("meta"):
        model = ViTPoseHuge()
    model.load_state_dict(data["state_dict"], strict=True, assign=True)
    model.source_config = data.get("meta", {}).get("cfg", "")
    return model.eval()


def crop_person(frame, box):
    """GetBBoxCenterScale(1.25) + TopdownAffine(use_udp=True), 192x256 input."""
    box = np.asarray(box, dtype=np.float32)
    center = (box[:2] + box[2:]) / 2
    width, height = (box[2:] - box[:2]) * 1.25
    if min(width, height) <= 0:
        raise ValueError("Empty golfer box")
    width = max(width, height * 192 / 256)
    height = max(height, width * 256 / 192)
    sx, sy = 191 / width, 255 / height
    matrix = np.array([[sx, 0, sx * (-center[0] + width / 2)],
                       [0, sy, sy * (-center[1] + height / 2)]], np.float32)
    crop = cv2.warpAffine(frame, matrix, (192, 256), flags=cv2.INTER_LINEAR)
    rgb = crop[..., ::-1].astype(np.float32)
    rgb = (rgb - np.array([123.675,116.28,103.53],np.float32)) / np.array([58.395,57.12,57.375],np.float32)
    return torch.from_numpy(rgb.transpose(2,0,1).copy())[None], center, np.array([width,height]), matrix


def decode_heatmaps(heatmaps, center, scale):
    heatmaps = np.array(heatmaps, dtype=np.float32, copy=True)
    points, scores = get_heatmap_maximum(heatmaps)
    points = refine_keypoints_dark_udp(points[None], heatmaps, blur_kernel_size=11)[0]
    # UDPHeatmap decode -> TopdownPoseEstimator inverse transform, matching MMPose.
    crop_xy = points / np.array([47,63]) * np.array([192,256])
    image_xy = crop_xy / np.array([192,256]) * scale + center - scale / 2
    return image_xy.astype(np.float32), scores.astype(np.float32)


def select_golfer(boxes, previous=None):
    if not len(boxes):
        return None
    areas = np.prod(np.maximum(boxes[:,2:] - boxes[:,:2], 0), axis=1)
    if previous is not None:
        inter = np.prod(np.maximum(np.minimum(boxes[:,2:],previous[2:]) - np.maximum(boxes[:,:2],previous[:2]),0),axis=1)
        prev_area = np.prod(np.maximum(previous[2:] - previous[:2],0))
        ious = inter / np.maximum(areas + prev_area - inter,1e-8)
        if ious.max() >= .1:
            return int(ious.argmax())
    return int(areas.argmax())


class ViTPoseTracking:
    def __init__(self, detector):
        self.detector = detector
        self.model = load_checkpoint()
        self.metadata = {"pose_backend":"vitpose-huge", "pose_checkpoint":str(CHECKPOINT.relative_to(ROOT)),
                         "person_detector":"yolo26n-pose.pt", "input_size_hw":[256,192],
                         "decoder":"UDPHeatmap + DARK refinement", "flip_test":True,
                         "video_association":"box IoU >= 0.1, otherwise largest golfer",
                         "strict_parameter_load":True}

    @property
    def predictor(self):
        return self.detector.predictor

    @torch.inference_mode()
    def _pose(self, detected, previous=None):
        from ultralytics.engine.results import Results
        frame = detected.orig_img
        boxes = detected.boxes.xyxy.cpu().numpy() if detected.boxes is not None else np.empty((0,4))
        index = select_golfer(boxes, previous)
        if index is None:
            return Results(frame,path="",names={0:"person"},boxes=torch.empty((0,6)),keypoints=torch.empty((0,17,3))), None
        box = boxes[index]
        start = perf_counter()
        image, center, scale, matrix = crop_person(frame,box)
        # Batch the original and mirror; all stochastic training layers are disabled.
        output = self.model(torch.cat((image,image.flip(-1))))
        heatmap = (output[0] + output[1].flip(-1)[COCO_FLIP]) / 2
        xy, conf = decode_heatmaps(heatmap.cpu().numpy(),center,scale)
        keypoints = np.concatenate((xy,conf[:,None]),axis=1)[None]
        score = float(detected.boxes.conf[index])
        result = Results(frame,path="",names={0:"person"},boxes=torch.tensor([[*box,score,0.]],dtype=torch.float32),
                         keypoints=torch.from_numpy(keypoints))
        result.vitpose_details = {"coco_xy":xy.tolist(),"coco_confidence":conf.tolist(),
                                 "crop_affine":matrix.tolist(),"inference_seconds":round(perf_counter()-start,3)}
        return result, box.copy()

    def predict(self, source, stream=False, **kwargs):
        results = self.detector.predict(source=source,stream=True,**kwargs)
        def iterate():
            previous = None
            for detected in results:
                result, box = self._pose(detected,previous)
                previous = box
                yield result
        generated = iterate()
        return generated if stream else list(generated)


def load_pose_backend(backend="yolo26n-pose", cpu_threads=4):
    from ultralytics import YOLO
    detector = YOLO(str(ROOT/"yolo26n-pose.pt"))
    if backend == "vitpose-huge":
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
        model = ViTPoseTracking(detector)
        model.metadata["cpu_threads"] = cpu_threads
        return model
    if backend != "yolo26n-pose":
        raise ValueError(f"Unknown pose backend: {backend}")
    detector.metadata = {"pose_backend":backend,"pose_checkpoint":"yolo26n-pose.pt","person_detector":"yolo26n-pose.pt"}
    return detector
