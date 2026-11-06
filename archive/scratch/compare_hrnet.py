import torch,timm,builtins,numpy as np
from golf_workflows.vitpose_tracking import _HistoryBuffer
from numpy._core import multiarray
allowed=[builtins.getattr,np.ndarray,np.dtype,(multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),(multiarray.scalar,'numpy.core.multiarray.scalar'),(_HistoryBuffer,'mmengine.logging.history_buffer.HistoryBuffer')]+[type(np.dtype(t)) for t in ('float32','float64','int64','int32','uint8','bool')]
with torch.serialization.safe_globals(allowed):
 s=torch.load('golfpose_checkpoints/golfpose_golfer_dekr.pth',weights_only=True,map_location='cpu',mmap=True)['state_dict']
m=timm.create_model('hrnet_w48',pretrained=False,features_only=True,feature_location='',out_indices=(1,2,3,4))
s={k.removeprefix('backbone.'):v for k,v in s.items() if k.startswith('backbone.')}
print('missing',set(m.state_dict())-set(s));print('extra',set(s)-set(m.state_dict()))
print(m.load_state_dict(s,strict=True))
