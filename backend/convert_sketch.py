"""Convert the official SDXS DreamShaper sketch pairing into one Core ML graph."""
import gc
from pathlib import Path
import torch

class FusedSketch(torch.nn.Module):
    def __init__(self,unet,controlnet):
        super().__init__()
        self.unet=unet
        self.controlnet=controlnet

    def forward(self,sample,timestep,encoder_hidden_states,control_image,control_strength):
        down,mid=self.controlnet(sample,timestep,encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_image,conditioning_scale=1.0,return_dict=False)
        down=tuple(value*control_strength for value in down)
        mid=mid*control_strength
        return self.unet(sample,timestep,encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down,mid_block_additional_residual=mid,
            return_dict=False)[0]

def main():
    import coremltools as ct
    import numpy as np
    from diffusers import UNet2DConditionModel,ControlNetModel
    from .model_specs import MODEL_CONFIGS,model_directory
    config=MODEL_CONFIGS['dreamshaper-sketch']
    folder=model_directory('dreamshaper-sketch')
    folder.mkdir(parents=True,exist_ok=True)
    # Reuse checkpoint-specific VAE packages already converted for DreamShaper.
    for name in ('taesd_encoder_512.mlpackage','taesd_decoder.mlpackage'):
        target=model_directory('dreamshaper')/name
        if not target.exists():raise FileNotFoundError(f'Convert DreamShaper first: {target}')
        destination=folder/name
        if not destination.exists():destination.symlink_to(target,target_is_directory=True)
    print('Loading DreamShaper UNet and sketch ControlNet on CPU',flush=True)
    unet=UNet2DConditionModel.from_pretrained(config['model_id'],subfolder='unet',torch_dtype=torch.float32).eval()
    controlnet=ControlNetModel.from_pretrained(config['controlnet_id'],torch_dtype=torch.float32).eval()
    model=FusedSketch(unet,controlnet).eval()
    examples=(torch.randn(1,4,64,64),torch.tensor([999.]),torch.randn(1,77,768),
              torch.zeros(1,3,512,512),torch.tensor([.8]))
    print('Tracing fused sketch graph',flush=True)
    with torch.no_grad():traced=torch.jit.trace(model,examples)
    print('Converting fused sketch graph to Core ML',flush=True)
    names=('sample','timestep','encoder_hidden_states','control_image','control_strength')
    converted=ct.convert(traced,inputs=[ct.TensorType(name=name,shape=value.shape,dtype=np.float16)
                           for name,value in zip(names,examples)],
             outputs=[ct.TensorType(name='noise_pred',dtype=np.float16)],
             compute_units=ct.ComputeUnit.CPU_AND_GPU,minimum_deployment_target=ct.target.macOS14,
             convert_to='mlprogram')
    path=folder/(config['unet_prefix']+'.mlpackage')
    converted.save(str(path))
    print(f'Saved {path}',flush=True)
    del converted,traced,model,unet,controlnet
    gc.collect()

if __name__=='__main__':main()
