import ast,copy,torch
from torch import nn
from torchvision.models.resnet import BasicBlock as TVBasic, Bottleneck as TVBottle
from golf_workflows.dekr_pose import load_dekr,resize_panel
import cv2
class Base(nn.Module):
 def __init__(self,init_cfg=None): super().__init__()
class Basic(TVBasic):
 def __init__(self,inc,outc,stride=1,downsample=None,**kwargs): super().__init__(inc,outc,stride,downsample)
class Bottle(TVBottle):
 def __init__(self,inc,outc,stride=1,downsample=None,**kwargs): super().__init__(inc,outc//4,stride,downsample)
def conv(cfg,*args,**kwargs): return nn.Conv2d(*args,**kwargs)
def norm(cfg,features,postfix=''): return 'bn'+str(postfix),nn.BatchNorm2d(features)
tree=ast.parse(open('.tmp/hrnet.py',encoding='utf8').read())
tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
for c in tree.body:c.decorator_list=[]
ns=dict(nn=nn,copy=copy,BaseModule=Base,BaseBackbone=Base,BasicBlock=Basic,Bottleneck=Bottle,get_expansion=lambda c:c.expansion,build_conv_layer=conv,build_norm_layer=norm,_BatchNorm=nn.modules.batchnorm._BatchNorm)
exec(compile(tree,'official_hrnet','exec'),ns)
extra={f'stage{i}':dict(num_modules=n,num_branches=i,block='BOTTLENECK' if i==1 else 'BASIC',num_blocks=(4,)*i,num_channels=(64,) if i==1 else tuple(48*2**j for j in range(i)),multiscale_output=True) for i,n in enumerate([1,1,4,3],1)}
torch.set_num_threads(8)
m=load_dekr()
r=ns['HRNet'](extra);r.eval();r.load_state_dict(m.backbone.state_dict(),strict=True)
x,*_=resize_panel(cv2.imread('.tmp/dekr_panel.jpg'))
with torch.inference_mode():
 a=m.backbone(x);b=r(x)
 print('diffs',[(u-v).abs().max().item() for u,v in zip(a,b)])
 f=torch.cat([b[0]]+[torch.nn.functional.interpolate(t,size=b[0].shape[-2:],mode='bilinear',align_corners=False) for t in b[1:]],1)
 hm,_=m.head(f);print('official stem max',hm[0,22].max().item())

