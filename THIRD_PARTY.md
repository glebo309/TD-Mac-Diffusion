# Third-party code and models

The root MIT license covers original project code. Third-party code retains its own license and attribution. Model weights are downloaded separately and are not relicensed by this repository.

## StreamDiffusion for Mac

`backend/vendor/ochyai_camera.py` and the starting point for `backend/convert_models.py` come from [ochyai/streamdiffusion-mac](https://github.com/ochyai/streamdiffusion-mac), retrieved 2026-10-06. Credit: Yoichi Ochiai (ochyai) and upstream contributors.

Upstream's README declares **MIT License**. No separate LICENSE file or copyright notice was present when reviewed. [The MIT text and attribution](licenses/StreamDiffusion-Mac-MIT.txt) are included here, with that provenance stated explicitly. This is an adaptation, not the unmodified upstream distribution.

Local changes include checkpoint configuration, DreamShaper text dimensions and VAE selection, prompt/seed controls, reset behavior, and conversion integration. The original camera application's display/capture routines are not used by the TouchDesigner worker.

## SDXS sketch example

The approach in `backend/sketch_pipeline.py` and `backend/convert_sketch.py` follows [IDKiro/sdxs/demo_sketch.py](https://github.com/IDKiro/sdxs/blob/main/demo_sketch.py), by the SDXS authors. That repository uses Apache License 2.0; its [license text](licenses/SDXS-Apache-2.0.txt) is included. Local adaptations add fused Core ML conversion, contour preprocessing, persistent streaming noise, and configurable feedback.

## Downloaded models

Consult the model cards and license files before downloading or distributing weights or derived model packages:

- [IDKiro/sdxs-512-0.9](https://huggingface.co/IDKiro/sdxs-512-0.9)
- [IDKiro/sdxs-512-dreamshaper](https://huggingface.co/IDKiro/sdxs-512-dreamshaper)
- [IDKiro/sdxs-512-dreamshaper-sketch](https://huggingface.co/IDKiro/sdxs-512-dreamshaper-sketch)
- [madebyollin/taesd](https://huggingface.co/madebyollin/taesd)
- [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo), experimental backend configuration only

DreamShaper conversion uses the TinyVAE bundled in the DreamShaper checkpoint. Model terms are independent of the MIT/Apache licenses on code. No checkpoint weights or converted Core ML packages are included in Git.

TouchDesigner is a separate Derivative product and must be installed under its own license. This project is an independent integration, not an official Derivative release.
