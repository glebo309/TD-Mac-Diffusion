"""Register the checkout used by the portable TouchDesigner component."""
from pathlib import Path


def register_install(project, home=None):
    project = Path(project).resolve()
    required = ('touchdesigner/bridge.py', 'worker/server.py', 'backend/pipeline.py')
    if not all((project / name).is_file() for name in required):
        raise ValueError(f'Incomplete Mac Diffusion checkout: {project}')
    config = (Path(home) if home is not None else Path.home()) / '.config/mac-diffusion/project.txt'
    config.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.with_suffix('.tmp')
    temporary.write_text(str(project) + '\n')
    temporary.replace(config)
    return config


if __name__ == '__main__':
    project = Path(__file__).resolve().parents[1]
    print(f'Registered {project} in {register_install(project)}')
