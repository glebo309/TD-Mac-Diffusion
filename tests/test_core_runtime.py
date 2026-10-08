from types import SimpleNamespace
from unittest.mock import Mock
import importlib

def load_runtime():
    from touchdesigner import core_runtime
    return importlib.reload(core_runtime)

def component(running):
    c=Mock()
    c.par=SimpleNamespace(Running=running,Prompt='glass sculpture',Seed=42,Feedback=.1,Wetdry=1,Model='sdxs',Shapestrength=.8,Lockseed=False)
    c.inputs=[object()]
    return c

def test_disabled_component_does_not_connect_or_launch_worker(monkeypatch):
    runtime=load_runtime()
    c=component(False)
    bridge=Mock()
    monkeypatch.setattr(runtime,'parent',lambda:c,raising=False)
    monkeypatch.setattr(runtime,'Bridge',bridge)
    runtime.start()
    runtime.tick()
    bridge.assert_not_called()
    c.op.assert_not_called()

def test_enabling_component_creates_bridge_lazily(monkeypatch):
    runtime=load_runtime()
    c=component(True)
    instance=Mock()
    instance.snapshot.return_value=(None,-1,{'state':'ready'})
    bridge=Mock(return_value=instance)
    monkeypatch.setattr(runtime,'parent',lambda:c,raising=False)
    monkeypatch.setattr(runtime,'Bridge',bridge)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:0)
    runtime.tick()
    bridge.assert_called_once()
    instance.submit.assert_called_with(None,{'running':True,'prompt':'glass sculpture','seed':42,'feedback':.1,'model':'sdxs','shape_strength':.8})

def test_disabling_sends_pause_without_reading_input(monkeypatch):
    runtime=load_runtime()
    c=component(False)
    runtime.client=Mock()
    monkeypatch.setattr(runtime,'parent',lambda:c,raising=False)
    runtime.tick()
    runtime.client.submit.assert_called_once_with(None,{'running':False,'prompt':'glass sculpture','seed':42,'feedback':.1,'model':'sdxs','shape_strength':.8})
    c.op.assert_not_called()

def test_generated_frame_stays_in_module_memory_not_component_storage(monkeypatch):
    import numpy as np
    runtime=load_runtime()
    c=component(True)
    instance=Mock()
    instance.snapshot.return_value=(np.zeros((512,512,3),dtype=np.uint8),1,{'state':'running'})
    monkeypatch.setattr(runtime,'parent',lambda:c,raising=False)
    monkeypatch.setattr(runtime,'Bridge',Mock(return_value=instance))
    monkeypatch.setattr(runtime.time,'monotonic',lambda:0)
    runtime.tick()
    assert not any(call.args[0]=='output_array' for call in c.store.call_args_list)
    assert runtime.output_array.shape==(512,512,3)


def test_model_selection_is_sent_by_both_runtimes(monkeypatch):
    import numpy as np
    for module_name in ('core_runtime','runtime'):
        runtime=importlib.reload(importlib.import_module('touchdesigner.'+module_name))
        c=component(True)
        c.par.Model='dreamshaper'
        c.op.return_value.numpyArray.return_value=np.zeros((512,512,4),dtype=np.float32)
        instance=Mock()
        instance.snapshot.return_value=(None,-1,{'state':'ready'})
        monkeypatch.setattr(runtime,'parent',lambda:c,raising=False)
        monkeypatch.setattr(runtime,'Bridge',Mock(return_value=instance))
        monkeypatch.setattr(runtime,'sync_panel',Mock(),raising=False)
        monkeypatch.setattr(runtime.time,'monotonic',lambda:1)
        runtime.last_status=1
        runtime.tick()
        assert instance.submit.call_args.args[1]['model']=='dreamshaper'


def test_shape_model_and_strength_are_sent_by_both_runtimes(monkeypatch):
    import numpy as np
    for module_name in ('core_runtime','runtime'):
        runtime=importlib.reload(importlib.import_module('touchdesigner.'+module_name))
        c=component(True)
        c.par.Model='dreamshaper-sketch'
        c.par.Shapestrength=.65
        c.op.return_value.numpyArray.return_value=np.zeros((512,512,4),dtype=np.float32)
        instance=Mock()
        instance.snapshot.return_value=(None,-1,{'state':'ready'})
        monkeypatch.setattr(runtime,'parent',lambda:c,raising=False)
        monkeypatch.setattr(runtime,'Bridge',Mock(return_value=instance))
        monkeypatch.setattr(runtime,'sync_panel',Mock(),raising=False)
        monkeypatch.setattr(runtime.time,'monotonic',lambda:1)
        runtime.last_status=1
        runtime.tick()
        sent=instance.submit.call_args.args[1]
        assert sent['model']=='dreamshaper-sketch'
        assert sent['shape_strength']==.65
