"""Optional GPU execution for the existing club network, retaining CPU I/O."""
import torch
from torch import nn


def resolve_devices(args):
    """Resolve the common preference once; explicit stage options win."""
    available = torch.cuda.is_available()
    preferred = args.device
    if preferred == 'auto':
        preferred = 'cuda' if available else 'cpu'
    for name in ('pose_device', 'club_device', 'aux_device', 'lifter_device'):
        value = getattr(args, name, None) or preferred
        if value == 'cuda' and not available:
            raise RuntimeError('CUDA unavailable; use --device cpu or install CUDA PyTorch with scripts/setup.ps1 -Cuda')
        setattr(args, name, value)
    return args


def move_detector(model, device):
    """Clear Ultralytics' cached predictor when crossing stage boundaries."""
    model.to(device)
    model.predictor = None
    model.overrides.update(device=device, half=False)


class StagedLifter(nn.Module):
    """Keep the shared CPU tensor API, loading MixSTE only when lifting starts."""
    def __init__(self, model, device, before_lift):
        super().__init__()
        self.model = model
        self.device = device
        self.before_lift = before_lift
        self.started = False
        self.eval()

    @property
    def Spatial_pos_embed(self):
        return self.model.Spatial_pos_embed

    @property
    def checkpoint_path(self):
        return self.model.checkpoint_path

    def forward(self, inputs):
        if not self.started:
            self.before_lift()
            self.model.to(self.device).float()
            self.started = True
        return self.model(inputs.to(self.device)).cpu()

    def release(self):
        self.model.to('cpu')
        self.started = False


class ClubDeviceModule(nn.Module):
    def __init__(self, module, device='cuda'):
        super().__init__()
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; install CUDA PyTorch with scripts/setup.ps1 -Cuda, or use --club-device cpu')
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.eval()

    def forward(self, images):
        # Keep the original two-image flip test and FP32 arithmetic.
        return self.module(images.to(self.device)).to('cpu')


def configure_club(club, device):
    if isinstance(club.model, ClubDeviceModule):
        club.model = club.model.module
    if device == 'cuda':
        club.model = ClubDeviceModule(club.model, device)
    club.metadata['inference_device'] = device
    club.metadata['precision'] = 'float32'
    return club
