"""Loopback-only raw RGB transport for TouchDesigner."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from .service import DiffusionService, FRAME_BYTES


class CoreMLEngine:
    def __init__(self, **options):
        from backend.pipeline import MacDiffusion
        self.engine = MacDiffusion(**options)

    def configure(self, **options):
        self.engine.configure(**options)

    def reset(self):
        self.engine.reset()

    def close(self):
        self.engine.close()

    def process_bytes(self, data):
        import numpy as np
        frame = np.frombuffer(data, dtype=np.uint8).reshape(512, 512, 3)
        return self.engine.process_frame(frame).astype(np.uint8).tobytes()


def make_server(service, port=8765):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, format, *args):
            # Frame polling would otherwise produce thousands of log lines.
            if args and str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
                super().log_message(format, *args)

        def send(self, code, data=b'', content_type='application/json', headers=None):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            if data:
                self.wfile.write(data)

        def json(self, code, value):
            self.send(code, json.dumps(value).encode())

        def is_local_request(self):
            hosts = {'127.0.0.1', 'localhost', '::1'}
            try:
                host = urlsplit('http://' + self.headers.get('Host', ''))
                valid = host.hostname in hosts and host.port == self.server.server_port
                origin = self.headers.get('Origin')
                if origin is not None:
                    parsed = urlsplit(origin)
                    valid = valid and parsed.scheme == 'http' and parsed.hostname in hosts
                    valid = valid and parsed.port == self.server.server_port
            except ValueError:
                valid = False
            if not valid:
                self.close_connection = True
                self.json(403, {'error': 'Only local worker requests are accepted'})
            return valid

        def do_GET(self):
            if not self.is_local_request():
                return
            if self.path == '/status':
                self.json(200, service.status())
            elif self.path == '/result':
                result = service.result()
                if result is None:
                    self.send(204)
                else:
                    frame_id, data, latency = result
                    self.send(200, data, 'application/octet-stream',
                              {'X-Frame-Id': frame_id, 'X-Latency-Ms': latency})
            else:
                self.json(404, {'error': 'Unknown endpoint'})

        def do_POST(self):
            if not self.is_local_request():
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                limit = FRAME_BYTES if self.path == '/frame' else 16384
                if not 0 < length <= limit:
                    self.close_connection = True
                    raise ValueError('Invalid body size')
                body = self.rfile.read(length)
                if self.path == '/frame':
                    self.json(202, {'id': service.submit(body)})
                elif self.path == '/control':
                    self.json(200, service.control(json.loads(body)))
                else:
                    self.json(404, {'error': 'Unknown endpoint'})
            except (ValueError, UnicodeError) as error:
                self.json(400, {'error': str(error)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    service = DiffusionService(CoreMLEngine)
    server = make_server(service, args.port)
    print(f'Mac Diffusion worker ready at http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
