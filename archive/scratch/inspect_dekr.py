from golf_workflows.vitpose_tracking import _HistoryBuffer
import torch, numpy as np, builtins, zipfile
from numpy._core import multiarray
allowed=[builtins.getattr,np.ndarray,np.dtype,(multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),(multiarray.scalar,'numpy.core.multiarray.scalar'),(_HistoryBuffer,'mmengine.logging.history_buffer.HistoryBuffer')]
allowed += [type(np.dtype(t)) for t in ('float32','float64','int64','int32','uint8','bool')]
with torch.serialization.safe_globals(allowed):
 d=torch.load('golfpose_checkpoints/golfpose_golfer_dekr.pth',map_location='cpu',weights_only=True,mmap=True)
s=d['state_dict']
print('KEYS',len(s))
print('\n'.join(f'{k} {tuple(v.shape)}' for k,v in s.items() if k.startswith('head.')))
print('BACKBONE',list(s)[:30])
