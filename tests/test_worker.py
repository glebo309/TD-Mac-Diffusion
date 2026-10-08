import threading
import time
import unittest

from worker.service import DiffusionService, FRAME_BYTES


def wait_for(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for worker')


class FakeEngine:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.frames = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def reset(self):
        pass

    def process_bytes(self, frame):
        self.frames.append(frame[0])
        self.entered.set()
        self.release.wait(1)
        return frame


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.service = DiffusionService(lambda **kw: self.engine)

    def tearDown(self):
        self.engine.release.set()
        self.service.close()

    def test_latest_frame_wins_without_backlog(self):
        self.service.control({'running': True, 'prompt': 'liquid chrome'})
        self.service.submit(bytes([1]) * FRAME_BYTES)
        self.assertTrue(self.engine.entered.wait(1))
        self.service.submit(bytes([2]) * FRAME_BYTES)
        last = self.service.submit(bytes([3]) * FRAME_BYTES)
        self.engine.release.set()
        wait_for(lambda: self.service.status()['generated_count'] == 2)
        self.assertEqual(self.engine.frames, [1, 3])
        self.assertEqual(self.service.result()[0], last)
        self.assertEqual(self.engine.options['prompt'], 'liquid chrome')

    def test_pause_keeps_output_and_stops_new_work(self):
        self.engine.release.set()
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['generated_count'] == 1)
        self.service.control({'running': False})
        self.service.submit(bytes([2]) * FRAME_BYTES)
        time.sleep(.05)
        self.assertEqual(self.service.status()['generated_count'], 1)
        self.assertIsNotNone(self.service.result())

    def test_invalid_control_is_atomic(self):
        before = self.service.status()
        with self.assertRaises(ValueError):
            self.service.control({'prompt': 'changed', 'feedback': 2})
        self.assertEqual(self.service.status()['prompt'], before['prompt'])
        with self.assertRaises(ValueError):
            self.service.control({'running': 'false'})

    def test_invalid_frame_rejected(self):
        with self.assertRaises(ValueError):
            self.service.submit(b'short')

    def test_inference_failure_is_visible_and_stops_worker(self):
        def fail(frame):
            raise RuntimeError('model unavailable')
        self.engine.process_bytes = fail
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['state'] == 'error')
        self.assertIn('model unavailable', self.service.status()['error'])
        self.assertFalse(self.service.status()['running'])

    def test_upstream_exit_is_visible_and_can_be_retried(self):
        attempts = []
        def factory(**kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise SystemExit(1)
            return self.engine
        self.service.close()
        self.service = DiffusionService(factory)
        self.engine.release.set()
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['state'] == 'error')
        self.assertIn('SystemExit', self.service.status()['error'])
        self.assertFalse(self.service.status()['running'])
        self.service.control({'running': True})
        wait_for(lambda: self.service.status()['generated_count'] == 1)

    def test_close_during_model_load_does_not_start_inference(self):
        loading = threading.Event()
        finish_loading = threading.Event()
        def factory(**kwargs):
            loading.set()
            finish_loading.wait(5)
            return self.engine
        self.service.close()
        self.service = DiffusionService(factory)
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        self.assertTrue(loading.wait(1))
        closer = threading.Thread(target=self.service.close)
        closer.start()
        wait_for(lambda: self.service._closed)
        finish_loading.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(self.engine.frames, [])

    def test_model_selection_validation_is_atomic(self):
        self.assertEqual(self.service.status()['model'], 'sdxs')
        self.service.control({'model': 'dreamshaper'})
        self.assertEqual(self.service.status()['model'], 'dreamshaper')
        with self.assertRaises(ValueError):
            self.service.control({'model': 'unknown', 'prompt': 'must not apply'})
        self.assertNotEqual(self.service.status()['prompt'], 'must not apply')
        self.assertEqual(self.service.status()['model'], 'dreamshaper')

    def test_switch_releases_old_engine_before_constructing_replacement(self):
        events = []
        engines = []
        def factory(**options):
            events.append(('load', options['model']))
            engine = FakeEngine(**options)
            engine.release.set()
            engine.close = lambda: events.append(('close', options['model']))
            engines.append(engine)
            return engine
        self.service.close()
        self.service = DiffusionService(factory)
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['generated_count'] == 1)
        self.service.control({'model': 'dreamshaper'})
        self.assertIsNone(self.service.result())
        self.assertEqual(self.service.status()['fps'], 0)
        self.service.submit(bytes([2]) * FRAME_BYTES)
        wait_for(lambda: self.service.status()['generated_count'] == 2)
        self.assertEqual(events[:3], [('load', 'sdxs'), ('close', 'sdxs'), ('load', 'dreamshaper')])
        self.assertEqual(engines[1].options['model'], 'dreamshaper')

    def test_model_switch_during_loading_never_processes_obsolete_model(self):
        loading = threading.Event()
        release = threading.Event()
        first = FakeEngine()
        second = FakeEngine()
        second.release.set()
        def factory(**options):
            if options['model'] == 'sdxs':
                loading.set()
                release.wait(2)
                return first
            return second
        self.service.close()
        self.service = DiffusionService(factory)
        try:
            self.service.control({'running': True})
            self.service.submit(bytes(FRAME_BYTES))
            self.assertTrue(loading.wait(1))
            self.service.control({'model': 'dreamshaper'})
            release.set()
            wait_for(lambda: self.service.status()['generated_count'] == 1)
            self.assertEqual(first.frames, [])
            self.assertEqual(second.frames, [0])
        finally:
            release.set()

    def test_switch_away_and_back_discards_inflight_result(self):
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        self.assertTrue(self.engine.entered.wait(1))
        self.service.control({'model': 'dreamshaper'})
        self.service.control({'model': 'sdxs'})
        self.engine.release.set()
        time.sleep(.05)
        self.assertIsNone(self.service.result())
        self.assertEqual(self.service.status()['generated_count'], 0)

    def test_obsolete_loading_failure_does_not_pause_selected_model(self):
        loading = threading.Event()
        release = threading.Event()
        def factory(**options):
            if options['model'] == 'sdxs':
                loading.set()
                release.wait(2)
                raise RuntimeError('old model failed')
            self.engine.release.set()
            return self.engine
        self.service.close()
        self.service = DiffusionService(factory)
        try:
            self.service.control({'running': True})
            self.service.submit(bytes(FRAME_BYTES))
            self.assertTrue(loading.wait(1))
            self.service.control({'model': 'dreamshaper'})
            release.set()
            wait_for(lambda: self.service.status()['generated_count'] == 1)
            self.assertTrue(self.service.status()['running'])
            self.assertEqual(self.service.status()['error'], '')
        finally:
            release.set()

    def test_worker_close_releases_loaded_engine_on_owner_thread(self):
        calls = []
        self.engine.close = lambda: calls.append(threading.current_thread().name)
        self.engine.release.set()
        self.service.control({'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['generated_count'] == 1)
        self.service.close()
        self.assertEqual(calls, ['diffusion'])

    def test_shape_strength_updates_live_without_reloading_sketch_model(self):
        loads = []
        def factory(**options):
            loads.append(options)
            return self.engine
        self.service.close()
        self.service = DiffusionService(factory)
        self.assertEqual(self.service.status()['shape_strength'], .8)
        self.engine.release.set()
        self.service.control({'model': 'dreamshaper-sketch', 'running': True})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['generated_count'] == 1)
        self.assertEqual(loads[0]['model'], 'dreamshaper-sketch')
        self.assertEqual(loads[0]['shape_strength'], .8)
        self.service.control({'shape_strength': .3})
        self.service.submit(bytes(FRAME_BYTES))
        wait_for(lambda: self.service.status()['generated_count'] == 2)
        self.assertEqual(len(loads), 1)
        self.assertEqual(self.engine.options['shape_strength'], .3)
        self.assertEqual(self.service.status()['shape_strength'], .3)

    def test_invalid_shape_strength_is_atomic(self):
        before = self.service.status()
        for value in (-.1, 1.01, float('nan'), float('inf'), True, '0.5', None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.service.control({'shape_strength': value, 'prompt': 'must not apply'})
                self.assertEqual(self.service.status(), before)
        for value in (0, 1):
            self.assertEqual(self.service.control({'shape_strength': value})['shape_strength'], value)

    def test_changed_prompt_discards_inflight_old_prompt_frame(self):
        self.service.control({'running': True, 'prompt': 'old prompt'})
        self.service.submit(bytes([1]) * FRAME_BYTES)
        self.assertTrue(self.engine.entered.wait(1))
        self.service.control({'prompt': 'new prompt'})
        frame_id = self.service.submit(bytes([2]) * FRAME_BYTES)
        self.engine.release.set()
        wait_for(lambda: self.service.result() is not None and self.service.result()[0] == frame_id)
        self.assertEqual(self.service.status()['generated_count'], 1)
        self.assertEqual(self.engine.options['prompt'], 'new prompt')

    def test_duplicate_controls_do_not_discard_inflight_frame(self):
        controls = {'running': True, 'prompt': 'same prompt'}
        self.service.control(controls)
        frame_id = self.service.submit(bytes(FRAME_BYTES))
        self.assertTrue(self.engine.entered.wait(1))
        self.service.control(controls)
        self.engine.release.set()
        wait_for(lambda: self.service.result() is not None)
        self.assertEqual(self.service.result()[0], frame_id)

    def test_shape_modulation_does_not_starve_inflight_output(self):
        self.service.control({'running': True, 'model': 'dreamshaper-sketch'})
        frame_id = self.service.submit(bytes(FRAME_BYTES))
        self.assertTrue(self.engine.entered.wait(1))
        self.service.control({'shape_strength': .4, 'feedback': .2})
        self.engine.release.set()
        wait_for(lambda: self.service.result() is not None)
        self.assertEqual(self.service.result()[0], frame_id)


if __name__ == '__main__':
    unittest.main()
