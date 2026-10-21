"""RGB boundary and live controls for the SDXS Core ML prototype.

One worker thread must own the instance. process_frame takes HxWx3 uint8
RGB, crops centrally, and returns 512x512 RGB uint8. The underlying pipeline
is ochyai's public webcam example, retained with provenance in vendor/.
"""
import numpy as np

from .model_specs import DEFAULT_MODEL_DIR, RESOLUTIONS, model_directory

def _release_memory():
    import gc
    import sys
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _create_pipeline(**kwargs):
    if kwargs['model_name'] == 'dreamshaper-sketch':
        from .sketch_pipeline import SketchPipeline
        return SketchPipeline(**kwargs)
    from .vendor.ochyai_camera import Pipeline
    return Pipeline(**kwargs)

class MacDiffusion:
    def __init__(self, prompt='colorful oil painting, expressive brushstrokes',
                 model_dir=None, seed=42, feedback=0.3, model='sdxs', shape_strength=0.8, resolution=512,
                 subject_edges=False):
        default_dir = model_directory(model)
        if resolution not in RESOLUTIONS:
            raise ValueError(f'Resolution must be one of {RESOLUTIONS}')
        # shortcut: sketch control and mask preprocessing are fixed at 512; convert per size to lift this.
        if model == 'dreamshaper-sketch':
            resolution = 512
        self.model = model
        self._validate(seed=seed, feedback=feedback, shape_strength=shape_strength)
        self._pipeline = _create_pipeline(
            model_name=model, render_size=resolution, output_size=512,
            prompt=str(prompt), latent_feedback=float(feedback),
            coreml_dir=str(model_dir or default_dir))
        self.seed = None
        self.configure(seed=seed, shape_strength=shape_strength, subject_edges=subject_edges)

    @staticmethod
    def _validate(seed=None, feedback=None, shape_strength=None):
        if seed is not None and (int(seed) != seed or not 0 <= seed <= 2**32-1):
            raise ValueError('Seed must be an integer from 0 to 4294967295')
        if shape_strength is not None and not 0 <= float(shape_strength) <= 1:
            raise ValueError('Shape strength must be between 0 and 1')
        if feedback is not None and not 0 <= float(feedback) <= 0.95:
            raise ValueError('Feedback must be between 0 and 0.95')

    def configure(self, prompt=None, seed=None, feedback=None, shape_strength=None, subject_edges=None):
        self._validate(seed=seed, feedback=feedback, shape_strength=shape_strength)
        p = self._pipeline
        if shape_strength is not None:
            p.shape_strength = float(shape_strength)
        if subject_edges is not None:
            p.subject_edges = bool(subject_edges)
        if prompt is not None and str(prompt) != p._current_prompt:
            embeddings = p._encode_single(str(prompt))
            p._target_embeds = embeddings
            p._prompt_embeds = embeddings.copy()
            p._current_prompt = str(prompt)
            self.reset()
        if seed is not None and int(seed) != self.seed:
            self.seed = int(seed)
            p._fixed_noise = np.random.RandomState(self.seed).randn(
                1, 4, p.latent_size, p.latent_size).astype(np.float16)
            self.reset()
        if feedback is not None:
            p.latent_feedback = float(feedback)

    def reset(self):
        self._pipeline._prev_denoised = None

    def close(self):
        if self._pipeline is not None:
            self._pipeline = None
            _release_memory()

    def process_frame(self, rgb):
        if self._pipeline is None:
            raise RuntimeError('Inference engine is closed')
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Frame must have shape height x width x 3')
        if rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
            raise ValueError('Frame must be a nonempty uint8 RGB image')
        bgr = self._pipeline.process_frame(np.ascontiguousarray(rgb[:, :, ::-1]))
        return np.ascontiguousarray(bgr[:, :, ::-1])
