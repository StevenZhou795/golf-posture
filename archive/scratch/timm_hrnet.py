import torch, timm
from golf_workflows.vitpose_tracking import _HistoryBuffer
m=timm.create_model('hrnet_w48',pretrained=False,features_only=True)
print(type(m),list(m.state_dict())[:25])
print(m)
