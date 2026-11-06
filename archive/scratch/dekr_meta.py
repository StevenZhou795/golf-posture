exec(open('.tmp/compare_hrnet.py',encoding='utf-8-sig').read().split("m=timm")[0])
with torch.serialization.safe_globals(allowed):
 d=torch.load('golfpose_checkpoints/golfpose_golfer_dekr.pth',weights_only=True,map_location='cpu',mmap=True)
print('meta',d['meta'].keys()); print('head meta', d['state_dict']._metadata.get('head'))
print(str(d['meta'].get('config',d['meta'].get('cfg','')))[:12000])
