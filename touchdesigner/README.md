# TouchDesigner component

Use **`dist/Mac Diffusion Core.tox`** after following the [project installation instructions](../README.md#install).

Connect one image TOP to its input and use its output in your network. Controls live on the **Diffusion** custom parameter page. Input is resized to 512 x 512; Generate off bypasses inference. Shape guidance requires the separately converted DreamShaper Shape model; output masking can use source alpha or brightness.

`core_runtime.py` is embedded in the component. `bridge.py` transports the latest RGB frame to a loopback worker from a background thread; TouchDesigner operator calls remain on the main thread. `controls.py` handles prompt changes and seed variations. `runtime.py` retains the earlier test-panel runtime for regression coverage; the public component does not require a panel.

`MacDiffusionTest.toe` is a test project that loads the built component from `../dist/`; see [Development](../README.md#development).

`component/` and `component.toc` contain the expanded operator network. Run `python scripts/build_component.py` from the repository to rebuild the distributable using TouchDesigner's `toecollapse` utility. No running project is changed by that build.

The component resolves its backend from `MAC_DIFFUSION_HOME`, the installation registered by `setup.sh`, or `~/Projects/TD-Mac-Diffusion`. Generated arrays live in module memory, not persistent operator storage. Use one enabled component at a time because the worker is shared.
