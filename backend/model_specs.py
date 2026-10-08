"""Checkpoint-specific dimensions shared by conversion and inference."""
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_CONFIGS = {
    'sdxs': {'model_id': 'IDKiro/sdxs-512-0.9', 'hidden_size': 1024,
             'unet_prefix': 'unet_sdxs_512', 'scheduler': 'euler'},
    'dreamshaper': {'model_id': 'IDKiro/sdxs-512-dreamshaper', 'hidden_size': 768,
                   'unet_prefix': 'unet_sdxs_dreamshaper_512', 'scheduler': 'euler',
                   'vae_model_id': 'IDKiro/sdxs-512-dreamshaper', 'vae_subfolder': 'vae'},
    'dreamshaper-sketch': {'model_id': 'IDKiro/sdxs-512-dreamshaper', 'hidden_size': 768,
                   'unet_prefix': 'unet_dreamshaper_sketch_512', 'scheduler': 'euler',
                   'controlnet_id': 'IDKiro/sdxs-512-dreamshaper-sketch',
                   'vae_model_id': 'IDKiro/sdxs-512-dreamshaper', 'vae_subfolder': 'vae'},
    'sd-turbo': {'model_id': 'stabilityai/sd-turbo', 'hidden_size': 1024,
                 'unet_prefix': 'unet_sd_turbo', 'scheduler': 'default'},
}

def model_directory(model):
    if model not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model: {model}')
    return DEFAULT_MODEL_DIR if model == 'sdxs' else DEFAULT_MODEL_DIR / model
