# Core ML backend

The [main README](../README.md) covers installation, models, measurements, and limitations.

- `model_specs.py`: model identifiers and checkpoint-specific dimensions.
- `convert_models.py`: SDXS/DreamShaper UNet and TinyVAE conversion.
- `convert_sketch.py`: fused DreamShaper UNet and sketch ControlNet conversion.
- `pipeline.py`: configurable inference adapter, seeds, and feedback resets.
- `sketch_pipeline.py`, `sketch.py`: sketch inference and contour preprocessing.
- `vendor/ochyai_camera.py`: adapted upstream Core ML pipeline; see [provenance](vendor/PROVENANCE.md).
- `benchmark.py`, `benchmark_sketch.py`: RAM-only frame processing with numerical JSON output.

SD-Turbo has an experimental backend configuration but is not exposed in the public component's model menu. SDXS and DreamShaper are the documented paths for this release.
