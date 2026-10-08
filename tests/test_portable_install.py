from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]


def runtime_project(monkeypatch, home):
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('MAC_DIFFUSION_HOME', raising=False)
    namespace = {}
    source = (ROOT / 'touchdesigner/core_runtime.py').read_text()
    exec(source.split('from bridge import Bridge')[0], namespace)
    return namespace['PROJECT']


def test_runtime_default_is_relative_to_current_user(monkeypatch, tmp_path):
    assert runtime_project(monkeypatch, tmp_path) == tmp_path / 'Projects/TD-Mac-Diffusion'


def test_runtime_uses_registered_install_in_folder_with_spaces(monkeypatch, tmp_path):
    config = tmp_path / '.config/mac-diffusion/project.txt'
    config.parent.mkdir(parents=True)
    install = tmp_path / 'Visual tools/Mac Diffusion'
    config.write_text(str(install) + '\n')
    assert runtime_project(monkeypatch, tmp_path) == install


def test_register_rejects_incomplete_project_without_changing_config(tmp_path):
    spec = importlib.util.spec_from_file_location('register', ROOT / 'scripts/register_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='Incomplete'):
        module.register_install(tmp_path / 'missing', home=tmp_path)
    assert not (tmp_path / '.config/mac-diffusion/project.txt').exists()


def test_register_records_real_project_location(tmp_path):
    spec = importlib.util.spec_from_file_location('register', ROOT / 'scripts/register_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.register_install(ROOT, home=tmp_path)
    assert config.read_text().strip() == str(ROOT)
