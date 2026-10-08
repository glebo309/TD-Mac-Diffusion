from backend import benchmark


def test_benchmark_retains_metrics_but_never_persists_frames(tmp_path, monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def process_frame(self, frame):
            return frame

    monkeypatch.setattr(benchmark, 'MacDiffusion', FakeEngine)
    monkeypatch.setattr(benchmark, 'OUT', tmp_path)
    benchmark.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['benchmark.json']
