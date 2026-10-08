import json
from backend import benchmark

def test_alternative_benchmark_records_correct_model_without_overwriting_default(tmp_path,monkeypatch):
    models=[]
    class FakeEngine:
        def __init__(self,**kwargs): models.append(kwargs['model'])
        def process_frame(self,frame): return frame
    monkeypatch.setattr(benchmark,'MacDiffusion',FakeEngine)
    monkeypatch.setattr(benchmark,'OUT',tmp_path)
    (tmp_path/'benchmark.json').write_text('original')
    benchmark.main(model='dreamshaper',frames=2)
    assert models==['dreamshaper']
    assert (tmp_path/'benchmark.json').read_text()=='original'
    metrics=json.loads((tmp_path/'benchmark-dreamshaper.json').read_text())
    assert metrics['model']=='IDKiro/sdxs-512-dreamshaper'
    assert metrics['frames']==2
    assert set(p.name for p in tmp_path.iterdir())=={'benchmark.json','benchmark-dreamshaper.json'}
