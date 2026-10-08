import numpy as np
import pytest
from backend import pipeline
from backend.sketch import prepare_control

def test_solid_shape_becomes_white_contour_on_black():
    rgb=np.zeros((512,512,3),dtype=np.uint8)
    rgb[160:352,160:352]=255
    control=prepare_control(rgb)
    assert control.shape==(1,3,512,512)
    assert control.dtype==np.float16
    assert control[0,0,250,250]==0
    assert control.max()==1
    assert np.count_nonzero(control[0,0])<5000
    assert control[0,0,155:165,150:360].max()==1

def test_preprocessing_preserves_position_and_is_invariant_to_polarity():
    rgb=np.zeros((512,512,3),dtype=np.uint8)
    rgb[80:200,50:180]=255
    a=prepare_control(rgb)
    b=prepare_control(255-rgb)
    assert np.array_equal(a,b)
    y,x=np.nonzero(a[0,0]);assert x.mean()<150 and y.mean()<180

def test_live_shape_strength_and_invalid_value(monkeypatch):
    class Fake:
        latent_size=64
        _current_prompt='shape'
        _prev_denoised=None
    fake=Fake()
    monkeypatch.setattr(pipeline,'_create_pipeline',lambda **kw:fake)
    engine=pipeline.MacDiffusion(prompt='shape',model='dreamshaper-sketch',shape_strength=.8)
    assert fake.shape_strength==.8
    engine.configure(shape_strength=.2)
    assert fake.shape_strength==.2
    with pytest.raises(ValueError,match='Shape'):
        engine.configure(shape_strength=1.2)
    assert fake.shape_strength==.2
