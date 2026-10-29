# Mac Diffusion for TouchDesigner

Local AI image processing for Apple Silicon, packaged as a reusable TouchDesigner component. Feed it a TOP from your own network and use the generated TOP downstream.

**Early development.** Tested with TouchDesigner 2025.32050 on an M4 Max with 32 GPU cores and 36 GB unified memory. Other Apple Silicon Macs are not yet validated. The focus is procedural shapes and live visuals, with a small parameter page rather than a standalone generation interface.

```text
Your TOP -> Mac Diffusion Core -> Generated TOP -> Your network
                |                       ^
                +-- local Core ML worker+
```

## What it does

- One image input and one image output at 512 x 512.
- SDXS 0.9, SDXS DreamShaper, and DreamShaper Shape with sketch ControlNet guidance.
- Prompt, seed, new variation, shape influence, latent feedback, and wet/dry controls.
- Optional output masking from input alpha or brightness, with transparent output outside the mask.
- Custom parameters usable from expressions and CHOP exports.
- Latest-frame transport in RAM. Generated images are overwritten, not recorded to disk.
- Generate off bypasses to the resized input. Loading a disabled core component does not launch inference.

## Requirements

- Apple Silicon Mac running macOS 14 or newer, without Rosetta.
- TouchDesigner; the bundled component was saved with 2025.32050.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) or Python 3.11.
- Internet access for the initial Python dependencies and model downloads.
- Several GB of free disk space for dependencies, source checkpoints, and converted models. Conversion temporarily uses more RAM and storage than inference.

## Install

```bash
git clone https://github.com/glebo309/TD-Mac-Diffusion.git
cd TD-Mac-Diffusion
bash setup.sh
.venv/bin/python -m backend.convert_models --model sdxs
```

Setup creates a project-local `.venv` and registers this checkout in `~/.config/mac-diffusion/project.txt`. You may clone anywhere, including a folder containing spaces. Keep the checkout and models in place after installing. If you move the folder, run `bash setup.sh --register-only` from its new location and reload the component. One installation is registered per macOS user; `MAC_DIFFUSION_HOME` can override that location for a process.

Model conversion is a separate, one-time operation and can take several minutes. It downloads the model from Hugging Face and creates Core ML packages locally. Model weights are not included in this repository. Their licenses apply separately; see [third-party notices](THIRD_PARTY.md).

## First use

1. Open a TouchDesigner project and drag **`dist/Mac Diffusion Core.tox`** into the network.
2. Connect a Noise TOP, rendered scene, movie, or other image TOP to its input.
3. Connect the component output to your downstream TOPs.
4. Select the component and open its **Diffusion** parameter page.
5. Keep **Model = SDXS 0.9**, enter a prompt, and turn **Generate** on. First loading and warmup take time.
6. For a fresh variation of the same prompt, pulse **New Variation**. Turn **Generate** off to stop inference and bypass the image.

The component starts the local worker automatically when enabled. It connects only to `127.0.0.1:8765`. Use one active Mac Diffusion component at a time in this prototype. TouchDesigner timeline pause is never overridden.

### Shape guidance and transparent output

For stronger control over generated forms, install the optional DreamShaper models:

```bash
.venv/bin/python -m backend.convert_models --model dreamshaper
.venv/bin/python -m backend.convert_sketch
```

Select **DreamShaper Shape** and adjust **Shape Influence**. Feed it clear silhouettes or contours; start with low feedback while tuning the input. Shape guidance influences the composition but does not promise exact object placement.

Enable **Mask to Input** when you want the generated output restricted to your source shape. Select **Brightness** for a white shape on black, or **Alpha** for a shape on transparency. Masking happens after generation. It crops the output to your shape; it is not semantic segmentation or diffusion inpainting.

Enable **Subject Edges** (DreamShaper Shape only) to guide generation from the main subject's outline instead of every edge in the frame. It uses macOS Vision subject detection and falls back to full-frame edges when no subject is found. It changes only the guidance; the output image is not masked.

## Parameters

| Parameter | Purpose |
|---|---|
| Generate | Enable inference; off bypasses to resized input |
| Model | Select an installed model; switching reloads the engine |
| Prompt | Text guidance; changing it clears previous-image feedback |
| Seed | Noise seed, held between frames for continuity |
| Lock Seed | Keep the seed on prompt changes; driven seeds also remain authoritative |
| New Variation | Choose a new seed for the current prompt |
| Shape Influence | Guidance strength for DreamShaper Shape only |
| Feedback | Blend previous generated latents into the next frame; 0 disables this history |
| Wet / Dry | Mix source and generated output |
| Mask to Input | Restrict output to the input mask |
| Mask Source | Use source alpha or brightness |
| Subject Edges | DreamShaper Shape only: take guide edges from the detected subject instead of the whole frame; off by default |
| Smoothing | Blend each new result into the previous output so playback looks continuous between inference frames; 0 shows raw frames. Stacks with latent Feedback |
| Resolution | Internal render size: 512, 384, or 256. Output stays 512 x 512; smaller is faster and softer. Not used by DreamShaper Shape |

## Performance and storage

Historical standalone measurements on the M4 Max above were roughly 13 inference FPS for SDXS/DreamShaper and 11.5 FPS for DreamShaper Shape at 512 x 512. These are model-processing measurements, not guaranteed TouchDesigner display FPS or end-to-end latency. TouchDesigner shares the GPU, and complex networks reduce available inference performance.

Inference loads with CPU and Neural Engine (`CPU_AND_NE`); prompt encoding stays on the GPU. Standalone SDXS on an M3 Pro measured about 12 FPS at 512, 20 at 384 and 38 at 256. Each smaller size is another converted UNet and VAE package, made once with, for example, `.venv/bin/python -m backend.convert_models --model sdxs --size 384`; selecting an unconverted size shows that command as the error.

Locally measured converted packages occupy approximately 631 MiB for SDXS, 607 MiB for DreamShaper, and 959 MiB for the sketch graph, which also reuses DreamShaper's VAE packages. Hugging Face source checkpoints, Python packages, and Core ML caches require additional space. Downloads may change over time.

Input and output frames stay in RAM; the worker retains the latest frames rather than building an image archive. `worker.log` contains diagnostics. Benchmark scripts save numerical JSON measurements only. Model weights and framework caches persist because they are needed for later runs.

## Troubleshooting

- **No output:** check that TouchDesigner is playing, Generate is on, the component has input, and the selected model was converted. The component never resumes your timeline itself.
- **Missing model package:** run the matching conversion command above. DreamShaper Shape requires DreamShaper conversion first.
- **Cannot import the bridge:** rerun `bash setup.sh --register-only` in the checkout and reload the component.
- **Old images influencing new ones:** set Feedback to 0, use New Variation, and inspect your upstream network for its own feedback. A held seed intentionally gives a stable result for unchanged input and prompt.
- **Worker connection problem:** inspect `worker.log`, or run `.venv/bin/python -m worker.server` in Terminal to see errors. Port 8765 must be available. Turn Generate off before troubleshooting.

## Development

The component's runtime source is `touchdesigner/core_runtime.py`; its expanded operator network is in `touchdesigner/component/`. Build the distributable with TouchDesigner's installed command-line utility:

```bash
.venv/bin/python scripts/build_component.py
```

Set `TOUCHDESIGNER_APP` if the app is installed somewhere other than `/Applications/TouchDesigner.app`. The build writes `dist/Mac Diffusion Core.tox`; it does not edit a running TouchDesigner project.

To try a build, open `touchdesigner/MacDiffusionTest.toe`. It feeds the component from a switch (`source_select` index 0 animated shape, 1 noise, 2 photo) and shows worker state, inference FPS and latency in `status_text`. The component references `../dist/Mac Diffusion Core.tox` instead of embedding a copy: after rebuilding, pulse **Enable External .tox Pulse** on `mac_diffusion`'s Common page (or reopen the project). A reload restores the component's parameter defaults, so turn **Generate** back on afterwards.

Run the tests without downloading or loading model weights:

```bash
uv pip install --python .venv/bin/python 'pytest>=9'
.venv/bin/python -m pytest tests backend -q
```

HTTP integration tests bind a temporary loopback port. A sandbox that forbids local sockets cannot run those seven tests. For an inference benchmark after converting models:

```bash
.venv/bin/python -m backend.benchmark --model sdxs
```

The current release is a prototype: one shared worker, fixed 512 x 512 output, no multi-component scheduling, and limited validation across Macs. Contributions and reproducible bug reports are welcome; include macOS, TouchDesigner version, chip/RAM, selected model, and relevant worker error.

## Credits and license

The Core ML backend adapts [ochyai/streamdiffusion-mac](https://github.com/ochyai/streamdiffusion-mac). Sketch conditioning follows the [SDXS authors' example](https://github.com/IDKiro/sdxs/blob/main/demo_sketch.py). This project adds TouchDesigner transport, parameters, prompt/seed handling, model switching, and output masking.

Original project code is MIT licensed. Upstream attribution and independent model licenses are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
