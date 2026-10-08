"""Rebuild the portable .tox offline with TouchDesigner's command-line tools."""
from pathlib import Path
import os
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    executable = Path(os.environ.get('TOUCHDESIGNER_APP', '/Applications/TouchDesigner.app')) / 'Contents/MacOS/toecollapse'
    if not executable.is_file():
        raise SystemExit('Install TouchDesigner or set TOUCHDESIGNER_APP to its .app path.')
    with tempfile.TemporaryDirectory(prefix='mac-diffusion-build-') as temporary:
        destination = Path(temporary) / 'Mac Diffusion Core.tox'
        expanded = Path(str(destination) + '.dir')
        shutil.copytree(ROOT / 'touchdesigner/component', expanded)
        shutil.copyfile(ROOT / 'touchdesigner/component.toc', str(destination) + '.toc')
        text = expanded / 'mac_diffusion_core/runtime.text'
        source = (ROOT / 'touchdesigner/core_runtime.py').read_bytes()
        text.write_bytes(text.read_bytes()[:23] + struct.pack('>I', len(source)) + source)
        # These utilities can return 1 on success. Validate the artifact as well.
        result = subprocess.run([str(executable), str(destination)], capture_output=True, text=True)
        if not destination.is_file() or destination.stat().st_size < 100:
            raise RuntimeError(result.stdout + result.stderr)
        target = ROOT / 'dist/Mac Diffusion Core.tox'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(destination, target)
        print(target)


if __name__ == '__main__':
    build()
