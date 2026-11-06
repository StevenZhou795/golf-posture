from golf_workflows.dekr_pose import *
torch.set_num_threads(8)
m=load_dekr()
p=cv2.imread('.tmp/dekr_panel.jpg')
with torch.inference_mode():
 for label,frame in [('full',p),('photo',p[89:])]:
  x,*_=resize_panel(frame)
  hm,dm=m(x)
  print(label, 'heatmaps',hm.shape)
  for j in [0,7,10,17,18,22]:
   h=hm[0,j]; i=int(h.argmax()); print(j, float(h.max()),(i%h.shape[1], i//h.shape[1]))
