"""Single-owner inference loop with bounded, latest-frame transport."""
import gc
import threading
import time

from backend.model_specs import RESOLUTIONS

FRAME_BYTES = 512 * 512 * 3
DEFAULT_PROMPT = 'iridescent liquid chrome sculpture, flowing organic forms, cinematic lighting'


class DiffusionService:
    def __init__(self, engine_factory):
        self._factory = engine_factory
        self._condition = threading.Condition()
        self._closed = False
        self._pending = None
        self._result = None
        self._sequence = 0
        self._revision = 0
        self._model_revision = 0
        self._content_revision = 0
        self._reset = 0
        self._status = dict(running=False, state='paused', model='sdxs', prompt=DEFAULT_PROMPT,
                            seed=42, feedback=0.1, shape_strength=0.8, subject_edges=False, resolution=512, fps=0.0, latency_ms=0.0,
                            generated_count=0, error='')
        self._thread = threading.Thread(target=self._run, daemon=True, name='diffusion')
        self._thread.start()

    def status(self):
        with self._condition:
            return dict(self._status)

    def control(self, values):
        if not isinstance(values, dict):
            raise ValueError('Control must be a JSON object')
        if set(values) - {'running', 'prompt', 'seed', 'feedback', 'reset', 'model', 'shape_strength', 'subject_edges', 'resolution'}:
            raise ValueError('Unknown control')
        if 'model' in values and values['model'] not in ('sdxs', 'dreamshaper', 'dreamshaper-sketch'):
            raise ValueError('Model must be sdxs, dreamshaper, or dreamshaper-sketch')
        if 'resolution' in values and (type(values['resolution']) is not int or values['resolution'] not in RESOLUTIONS):
            raise ValueError(f'Resolution must be one of {RESOLUTIONS}')
        if 'prompt' in values and (not isinstance(values['prompt'], str) or len(values['prompt']) > 2000):
            raise ValueError('Prompt must be text, at most 2000 characters')
        if 'seed' in values and (type(values['seed']) is not int or not 0 <= values['seed'] <= 4294967295):
            raise ValueError('Seed must be an integer from 0 to 4294967295')
        if 'feedback' in values and (type(values['feedback']) not in (int, float) or not 0 <= values['feedback'] <= .95):
            raise ValueError('Feedback must be between 0 and 0.95')
        if 'shape_strength' in values and (type(values['shape_strength']) not in (int, float) or not 0 <= values['shape_strength'] <= 1):
            raise ValueError('Shape strength must be between 0 and 1')
        for key in ('running', 'reset', 'subject_edges'):
            if key in values and type(values[key]) is not bool:
                raise ValueError(f'{key} must be a boolean')
        with self._condition:
            # DreamShaper Shape only renders at 512; a hidden menu value must not reload it every tick.
            if values.get('model', self._status['model']) == 'dreamshaper-sketch':
                values = {**values, 'resolution': 512}
            # A new render size needs a new engine, exactly like a model switch.
            changed_model = any(values.get(key, self._status[key]) != self._status[key] for key in ('model', 'resolution'))
            if changed_model or values.get('prompt', self._status['prompt']) != self._status['prompt']:
                self._content_revision += 1
            changed_generation = values.get('reset', False) or any(
                key in values and values[key] != self._status[key]
                for key in ('model', 'resolution', 'prompt', 'seed', 'feedback', 'shape_strength', 'subject_edges'))
            if changed_model:
                self._model_revision += 1
                self._result = None
                self._status.update(fps=0.0, latency_ms=0.0, error='')
            self._status.update({k: v for k, v in values.items() if k != 'reset'})
            self._revision += int(changed_generation)
            self._reset += int(values.get('reset', False))
            if values.get('running') or (changed_model and self._status['running']):
                self._status['error'] = ''
                self._status['state'] = 'starting'
            elif values.get('running') is False:
                self._status['state'] = 'paused'
            self._condition.notify_all()
            return dict(self._status)

    def submit(self, frame):
        if len(frame) != FRAME_BYTES:
            raise ValueError(f'Expected {FRAME_BYTES} raw RGB bytes (512 x 512)')
        with self._condition:
            self._sequence += 1
            self._pending = (self._sequence, frame, time.monotonic())
            self._condition.notify_all()
            return self._sequence

    def result(self):
        with self._condition:
            return self._result

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=2)

    def _run(self):
        engine = None
        engine_model_revision = -1
        revision = -1
        reset = 0
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or (self._status['running'] and self._pending is not None))
                    if self._closed:
                        return
                    current_revision = self._revision
                    current_model_revision = self._model_revision
                    current_content_revision = self._content_revision
                    model = self._status['model']
                    resolution = self._status['resolution']
                    current_reset = self._reset
                    options = {k: self._status[k] for k in ('prompt', 'seed', 'feedback', 'shape_strength', 'subject_edges')}
                    self._status['state'] = 'loading' if engine is None or engine_model_revision != current_model_revision else 'running'
                try:
                    if engine is not None and engine_model_revision != current_model_revision:
                        if hasattr(engine, 'close'):
                            engine.close()
                        engine = None
                        gc.collect()
                    if engine is None:
                        engine = self._factory(model=model, resolution=resolution, **options)
                        engine_model_revision = current_model_revision
                        revision = -1
                    if revision != current_revision:
                        engine.configure(**options)
                        revision = current_revision
                    if reset != current_reset:
                        engine.reset()
                        reset = current_reset
                    # Fetch AFTER model loading so startup does not render an old frame.
                    with self._condition:
                        if self._closed:
                            return
                        if not self._status['running'] or current_revision != self._revision:
                            continue
                        frame_id, data, submitted = self._pending
                        self._pending = None
                        self._status['state'] = 'running'
                    started = time.monotonic()
                    output = engine.process_bytes(data)
                    finished = time.monotonic()
                    if len(output) != FRAME_BYTES:
                        raise ValueError('Inference returned an invalid frame size')
                    with self._condition:
                        # Continuous slider modulation must not starve output;
                        # only superseded prompt/model content is discarded.
                        if self._closed or current_content_revision != self._content_revision:
                            continue
                        latency = (finished - submitted) * 1000
                        self._result = (frame_id, output, latency)
                        self._status['latency_ms'] = latency
                        self._status['fps'] = 1 / max(finished - started, 1e-9)
                        self._status['generated_count'] += 1
                except (Exception, SystemExit) as error:
                    with self._condition:
                        if current_revision != self._revision:
                            continue
                        self._status.update(running=False, state='error', error=f'{type(error).__name__}: {error}')
        finally:
            if engine is not None and hasattr(engine, 'close'):
                engine.close()

