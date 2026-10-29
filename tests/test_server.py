import json
import threading
import urllib.error
import urllib.request
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from worker.server import make_server, CoreMLEngine
from worker.service import DiffusionService, FRAME_BYTES
from test_worker import FakeEngine, wait_for


class EngineAdapterTests(unittest.TestCase):
    def test_model_and_close_are_forwarded_to_backend(self):
        calls = []
        backend = SimpleNamespace(close=lambda: calls.append('close'))
        def factory(**options):
            calls.append(options)
            return backend
        with patch.dict('sys.modules', {'backend.pipeline': SimpleNamespace(MacDiffusion=factory)}):
            engine = CoreMLEngine(model='dreamshaper', seed=12)
        engine.close()
        self.assertEqual(calls, [{'model': 'dreamshaper', 'seed': 12}, 'close'])


class EngineOutputTests(unittest.TestCase):
    def test_result_is_rgb_only(self):
        engine = CoreMLEngine.__new__(CoreMLEngine)
        engine.engine = SimpleNamespace(process_frame=lambda f: f)
        self.assertEqual(engine.process_bytes(bytes([9]) * FRAME_BYTES), bytes([9]) * FRAME_BYTES)


class HttpTests(unittest.TestCase):
    def setUp(self):
        engine = FakeEngine()
        engine.release.set()
        self.service = DiffusionService(lambda **kw: engine)
        self.server = make_server(self.service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.service.close()

    def request(self, path, data=None):
        return urllib.request.urlopen(self.url + path, data=data, timeout=2)

    def test_end_to_end_frame_and_controls(self):
        self.assertEqual(self.request('/result').status, 204)
        response = self.request('/control', json.dumps({'running': True}).encode())
        self.assertTrue(json.load(response)['running'])
        frame = bytes([128]) * FRAME_BYTES
        self.assertEqual(self.request('/frame', frame).status, 202)
        wait_for(lambda: self.service.result() is not None)
        with self.request('/result') as response:
            self.assertEqual(response.read(), frame)
            self.assertEqual(response.headers['X-Frame-Id'], '1')
        self.assertEqual(json.load(self.request('/status'))['generated_count'], 1)

    def test_bad_request_has_meaningful_error(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request('/control', b'{"seed":-1}')
        self.assertEqual(raised.exception.code, 400)
        self.assertIn('Seed', json.load(raised.exception)['error'])

    def test_sketch_controls_round_trip_and_invalid_strength_returns_400(self):
        controls = {'model': 'dreamshaper-sketch', 'shape_strength': .65}
        with self.request('/control', json.dumps(controls).encode()) as response:
            status = json.load(response)
        self.assertEqual(status['model'], controls['model'])
        self.assertEqual(status['shape_strength'], .65)
        self.assertEqual(json.load(self.request('/status'))['shape_strength'], .65)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request('/control', b'{"shape_strength": 1.5}')
        self.assertEqual(raised.exception.code, 400)
        self.assertIn('Shape strength', json.load(raised.exception)['error'])

    def test_foreign_origin_cannot_change_controls(self):
        request = urllib.request.Request(
            self.url + '/control', data=b'{"running": true}',
            headers={'Origin': 'https://unrelated.example', 'Content-Type': 'text/plain'})
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 403)
        self.assertFalse(self.service.status()['running'])

    def test_foreign_host_cannot_read_status(self):
        request = urllib.request.Request(self.url + '/status', headers={'Host': 'attacker.example'})
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 403)

    def test_local_origin_is_accepted(self):
        request = urllib.request.Request(
            self.url + '/control', data=b'{"running": true}',
            headers={'Origin': self.url})
        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertTrue(json.load(response)['running'])

    def test_null_origin_cannot_change_controls(self):
        request = urllib.request.Request(
            self.url + '/control', data=b'{"running": true}',
            headers={'Origin': 'null'})
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(raised.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
