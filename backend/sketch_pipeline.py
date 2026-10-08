"""One-step SDXS sketch ControlNet, adapted from IDKiro/sdxs demo_sketch.py."""
import cv2
import numpy as np
from .vendor.ochyai_camera import Pipeline
from .sketch import prepare_control

class SketchPipeline(Pipeline):
    def __init__(self,*args,**kwargs):
        self.shape_strength=.8
        super().__init__(*args,**kwargs)

    def _warmup(self,n=12):
        guide=np.zeros((512,512,3),dtype=np.uint8)
        for _ in range(n):self.process_frame(guide)
        self._prev_denoised=None

    def process_frame(self,frame_bgr):
        control=prepare_control(np.ascontiguousarray(frame_bgr[:,:,::-1]))
        # Euler t=999 input scaling and one-step x0 reconstruction. Starting
        # with fixed noise matches upstream text-to-image sketch conditioning.
        noisy=self._sqrt_1ma*self._fixed_noise
        if self._prev_denoised is not None and self.latent_feedback>0:
            noisy=noisy+self._sqrt_a*np.float16(self.latent_feedback)*self._prev_denoised
        predicted=self.unet.predict({'sample':noisy.astype(np.float16),
            'timestep':self._t_buf,'encoder_hidden_states':self._prompt_embeds,
            'control_image':control,'control_strength':np.array([self.shape_strength],dtype=np.float16)})
        denoised=(noisy-self._sqrt_1ma*np.asarray(predicted['noise_pred'],dtype=np.float16))/self._sqrt_a
        self._prev_denoised=denoised.copy()
        decoded=self.vae_decoder.predict({'latent':denoised})['image']
        rgb=np.asarray(decoded,dtype=np.float32).squeeze(0).transpose(1,2,0)
        rgb=((rgb+1)*127.5).clip(0,255).astype(np.uint8)
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
