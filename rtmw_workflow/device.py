"""Optional GPU execution for the existing club network, retaining CPU I/O."""
import torch
from torch import nn


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
