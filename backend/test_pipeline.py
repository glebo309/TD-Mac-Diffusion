import numpy as np
import pytest
from backend import pipeline

class FakePipeline:
    latent_size = 64
    _current_prompt = 'initial'
    _prompt_embeds = np.zeros((1, 77, 1024), dtype=np.float16)
    _prev_denoised = 'old'
    latent_feedback = 0.3

    def _encode_single(self, prompt):
        self.encoded = prompt
        return np.ones_like(self._prompt_embeds)

    def process_frame(self, bgr):
        self.received = bgr.copy()
        return bgr

@pytest.fixture
def engine(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(pipeline, '_create_pipeline', lambda **kwargs: fake)
    obj = pipeline.MacDiffusion(prompt='initial')
    return obj, fake

def test_rgb_boundary(engine):
    obj, fake = engine
    rgb = np.full((512, 512, 3), [255, 50, 0], dtype=np.uint8)
    result = obj.process_frame(rgb)
    assert tuple(fake.received[0, 0]) == (0, 50, 255)
    assert np.array_equal(result, rgb)
    assert result.flags.c_contiguous

def test_control_updates_change_generation_state(engine):
    obj, fake = engine
    obj.configure(prompt='new prompt', seed=123, feedback=0.6)
    assert fake.encoded == 'new prompt'
    assert np.all(fake._target_embeds == 1)
    assert fake.latent_feedback == 0.6
    expected = np.random.RandomState(123).randn(1,4,64,64).astype(np.float16)
    assert np.array_equal(fake._fixed_noise, expected)
    assert fake._prev_denoised is None

def test_invalid_control_rejected_without_partial_mutation(engine):
    obj, fake = engine
    with pytest.raises(ValueError):
        obj.configure(prompt='should not apply', feedback=2)
    assert fake._current_prompt == 'initial'
    with pytest.raises(ValueError):
        obj.process_frame(np.zeros((10,10), dtype=np.uint8))

def test_model_selection_uses_separate_cache_and_preserves_default(monkeypatch):
    created=[]
    monkeypatch.setattr(pipeline, '_create_pipeline', lambda **kw: created.append(kw) or FakePipeline())
    pipeline.MacDiffusion()
    pipeline.MacDiffusion(model='dreamshaper')
    assert created[0]['model_name']=='sdxs'
    assert created[0]['coreml_dir']==str(pipeline.DEFAULT_MODEL_DIR)
    assert created[1]['model_name']=='dreamshaper'
    assert created[1]['coreml_dir']==str(pipeline.DEFAULT_MODEL_DIR/'dreamshaper')
    with pytest.raises(ValueError,match='Unknown model'):
        pipeline.MacDiffusion(model='unknown')
    assert len(created)==2

def test_model_specs_match_checkpoint_embedding_dimensions():
    from backend.model_specs import MODEL_CONFIGS
    assert MODEL_CONFIGS['dreamshaper']['hidden_size']==768
    assert MODEL_CONFIGS['sdxs']['hidden_size']==1024
    assert MODEL_CONFIGS['dreamshaper']['model_id']=='IDKiro/sdxs-512-dreamshaper'
    assert MODEL_CONFIGS['dreamshaper']['scheduler']=='euler'

def test_close_releases_pipeline_and_prevents_reuse(engine, monkeypatch):
    obj, fake = engine
    released=[]
    monkeypatch.setattr(pipeline,'_release_memory',lambda:released.append(True))
    obj.close()
    obj.close()
    assert obj._pipeline is None
    assert released==[True]
    with pytest.raises(RuntimeError,match='closed'):
        obj.process_frame(np.zeros((512,512,3),dtype=np.uint8))

def test_dreamshaper_uses_its_bundled_decoder_weights():
    from backend.model_specs import MODEL_CONFIGS
    config=MODEL_CONFIGS['dreamshaper']
    assert config['vae_model_id']=='IDKiro/sdxs-512-dreamshaper'
    assert config['vae_subfolder']=='vae'

def test_prompt_change_starts_clean_without_old_prompt_or_image(engine):
    obj, fake = engine
    fake._prompt_embeds = np.zeros((1, 77, 1024), dtype=np.float16)
    fake._prev_denoised = np.full((1, 4, 64, 64), 17, dtype=np.float16)
    noise = fake._fixed_noise.copy()
    obj.configure(prompt='a flower', feedback=.14)
    assert np.array_equal(fake._prompt_embeds, fake._target_embeds)
    assert fake._prev_denoised is None
    assert fake.latent_feedback == .14
    assert np.array_equal(fake._fixed_noise, noise)


def test_unchanged_prompt_preserves_intentional_feedback(engine):
    obj, fake = engine
    previous = np.full((1, 4, 64, 64), 17, dtype=np.float16)
    fake._prev_denoised = previous
    obj.configure(prompt='initial', feedback=.14)
    assert fake._prev_denoised is previous
