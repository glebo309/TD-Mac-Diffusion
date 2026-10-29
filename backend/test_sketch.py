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

def _noise():
    return (np.random.RandomState(0).rand(512,512,3)*255).astype(np.uint8)

def test_subject_mask_gates_edges_outside_subject(monkeypatch):
    import backend.sketch as sketch
    mask=np.zeros((512,512),dtype=np.uint8);mask[100:400,150:350]=255
    monkeypatch.setattr(sketch,'_subject_mask',lambda rgb:mask)
    control,returned=sketch.prepare_control_and_mask(_noise(),subject_edges=True)
    assert returned.shape==(512,512) and returned[200,200]==255 and returned[10,10]==0
    assert control[0,0,10:90,10:90].max()==0  # no edges outside the subject

def test_subject_edges_default_off_skips_vision(monkeypatch):
    import backend.sketch as sketch
    def boom(rgb):raise AssertionError('Vision called')
    monkeypatch.setattr(sketch,'_subject_mask',boom)
    control,returned=sketch.prepare_control_and_mask(_noise())
    assert returned is None and control[0,0,10:90,10:90].max()==1

@pytest.mark.parametrize('mask',[None,np.zeros((512,512),dtype=np.uint8),
                                 np.pad(np.full((30,30),255,dtype=np.uint8),((0,482),(0,482)))])
def test_missing_empty_or_tiny_mask_uses_full_frame_edges(monkeypatch,mask):
    import backend.sketch as sketch
    monkeypatch.setattr(sketch,'_subject_mask',lambda rgb:mask)
    control,returned=sketch.prepare_control_and_mask(_noise(),subject_edges=True)
    assert returned is None
    assert np.array_equal(control,sketch.prepare_control(_noise()))

def test_no_vision_instances_is_no_mask(monkeypatch):
    import sys,types
    import backend.sketch as sketch
    class Request:
        @classmethod
        def alloc(cls):return cls()
        def init(self):return self
        def results(self):return []
    class Handler:
        def performRequests_error_(self,requests,error):return True,None
    monkeypatch.setitem(sys.modules,'Vision',types.SimpleNamespace(VNGenerateForegroundInstanceMaskRequest=Request))
    monkeypatch.setattr(sketch,'_vision_handler',lambda rgb:Handler())
    assert sketch._subject_mask(_noise()) is None
