import builtins,zipfile,json
from pathlib import Path
import torch,numpy as np
from numpy._core import multiarray
from golf_workflows.vitpose_tracking import _HistoryBuffer
allowed=[builtins.getattr,np.ndarray,np.dtype,(multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),(multiarray.scalar,'numpy.core.multiarray.scalar'),(_HistoryBuffer,'mmengine.logging.history_buffer.HistoryBuffer')]
allowed += [type(np.dtype(t)) for t in ('float32','float64','int64','int32','uint8','bool')]
for count in (1,2):
 p=Path(f'golfpose_checkpoints/golfpose_detector_{count}cls_yolox_s.pth')
 with torch.serialization.safe_globals(allowed):
  d=torch.load(p,map_location='cpu',weights_only=True,mmap=zipfile.is_zipfile(p))
 meta=d.get('meta',{})
 print(p.name,'meta keys',list(meta),'headshape',d['state_dict']['bbox_head.multi_level_conv_cls.0.weight'].shape)
 print('classes',meta.get('dataset_meta',meta.get('CLASSES')))
 cfg=meta.get('cfg',meta.get('config',''))
 Path(f'.tmp/detector_{count}cls_config.txt').write_text(str(cfg),encoding='utf-8')
 print(str(cfg)[:1500])
