"""Image contours in the white-on-black convention of SDXS sketch ControlNet."""
import cv2
import numpy as np

def prepare_control(rgb):
    h,w=rgb.shape[:2]
    side=min(h,w)
    rgb=rgb[(h-side)//2:(h+side)//2,(w-side)//2:(w+side)//2]
    rgb=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,64,128)
    edges=cv2.dilate(edges,np.ones((3,3),dtype=np.uint8))
    image=np.repeat((edges.astype(np.float16)/255)[None,None],3,axis=1)
    return np.ascontiguousarray(image)
