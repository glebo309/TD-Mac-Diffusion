import torch
from backend.convert_sketch import FusedSketch

class FakeControl(torch.nn.Module):
    def forward(self,sample,timestep,encoder_hidden_states,controlnet_cond,conditioning_scale,return_dict):
        assert conditioning_scale==1.0
        return (controlnet_cond, controlnet_cond*2), controlnet_cond*3

class FakeUNet(torch.nn.Module):
    def forward(self,sample,timestep,encoder_hidden_states,down_block_additional_residuals,mid_block_additional_residual,return_dict):
        return (sample+sum(down_block_additional_residuals)+mid_block_additional_residual,)

def test_fused_sketch_scales_all_residuals_and_zero_disables_conditioning():
    model=FusedSketch(FakeUNet(),FakeControl())
    x=torch.ones(1); guide=torch.ones(1)*2
    assert torch.equal(model(x,x,x,guide,torch.tensor([0.])),x)
    assert torch.equal(model(x,x,x,guide,torch.tensor([.5])),torch.tensor([7.]))
