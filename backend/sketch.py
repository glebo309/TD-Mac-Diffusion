"""Image contours in the white-on-black convention of SDXS sketch ControlNet."""
import cv2
import numpy as np

def _canny(gray):
    edges=cv2.Canny(gray,64,128)
    return cv2.dilate(edges,np.ones((3,3),dtype=np.uint8))

def _read_mask(pixel,dtype):
    from Quartz.CoreVideo import (
        CVPixelBufferGetBaseAddress,CVPixelBufferGetBytesPerRow,CVPixelBufferGetHeight,
        CVPixelBufferGetWidth,CVPixelBufferLockBaseAddress,CVPixelBufferUnlockBaseAddress)
    CVPixelBufferLockBaseAddress(pixel,0)
    try:
        height,width=CVPixelBufferGetHeight(pixel),CVPixelBufferGetWidth(pixel)
        stride=CVPixelBufferGetBytesPerRow(pixel)
        raw=np.frombuffer(CVPixelBufferGetBaseAddress(pixel).as_buffer(stride*height),dtype=dtype)
        mask=raw.reshape(height,stride//raw.itemsize)[:,:width].copy()
    finally:
        CVPixelBufferUnlockBaseAddress(pixel,0)
    return mask if dtype==np.uint8 else (mask*255).clip(0,255).astype(np.uint8)

def _vision_handler(rgb):
    from Foundation import NSData
    import Vision
    ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY),80])
    if not ok:
        return None
    data=NSData.dataWithBytes_length_(encoded.tobytes(),len(encoded))
    return Vision.VNImageRequestHandler.alloc().initWithData_options_(data,None)

def _subject_mask(rgb):
    """People, animals and other salient subjects (Vision subject lifting, macOS 14+).
    None when Vision is unavailable or the frame has no subject."""
    try:
        import Vision
        handler=_vision_handler(rgb)
        request=Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
    except Exception:
        return None
    if handler is None:
        return None
    success,_error=handler.performRequests_error_([request],None)
    if not success:
        return None
    results=request.results()
    if not results:
        return None
    pixel,_error=results[0].generateScaledMaskForImageForInstances_fromRequestHandler_error_(
        results[0].allInstances(),handler,None)
    return None if pixel is None else _read_mask(pixel,np.float32)

def prepare_control(rgb,subject_edges=False):
    return prepare_control_and_mask(rgb,subject_edges)[0]

def prepare_control_and_mask(rgb,subject_edges=False):
    """Returns (control image, 512x512 uint8 subject mask or None).
    None means full-frame edges: gating off, Vision unavailable, or no subject over 0.5% of the frame."""
    h,w=rgb.shape[:2]
    side=min(h,w)
    rgb=rgb[(h-side)//2:(h+side)//2,(w-side)//2:(w+side)//2]
    rgb=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    mask=_subject_mask(rgb) if subject_edges else None
    if mask is not None:
        mask=cv2.resize(mask,(512,512),interpolation=cv2.INTER_LINEAR)
        binary=(mask>128).astype(np.uint8)*255
        if cv2.countNonZero(binary)<=512*512*0.005:
            mask=None
    if mask is None:
        edges=_canny(gray)
    else:
        edges=_canny(cv2.bitwise_and(gray,binary))
        contours,_hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges,contours,-1,255,4)
    image=np.repeat((edges.astype(np.float16)/255)[None,None],3,axis=1)
    return np.ascontiguousarray(image),mask
