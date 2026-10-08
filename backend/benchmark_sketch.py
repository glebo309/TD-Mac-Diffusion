"""Sketch acceptance experiment. Frames remain in RAM; JSON metrics only on disk."""
import argparse
import io
import json
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .pipeline import MacDiffusion

ROOT=Path(__file__).resolve().parents[1]

def guide(kind):
    image=Image.new('RGB',(512,512),'black');draw=ImageDraw.Draw(image)
    if kind<2:
        x=155 if kind==0 else 350
        draw.ellipse((x-85,165,x+85,335),fill='white')
    else:
        draw.line([(80,360),(180,170),(285,340),(410,145)],fill='white',width=44,joint='curve')
    return np.asarray(image)

def main(preview_port=None):
    engine=MacDiffusion(model='dreamshaper-sketch',prompt='a single liquid chrome sculpture, studio photograph, black background',feedback=0,shape_strength=0)
    guides=[guide(i) for i in range(3)]
    no_control=[engine.process_frame(item) for item in guides]
    zero_delta=float(np.abs(no_control[0].astype(float)-no_control[1]).mean())
    assert zero_delta<.1,f'Disabled control still changes output: {zero_delta}'
    engine.configure(shape_strength=.8)
    controlled=[engine.process_frame(item) for item in guides]
    strong_delta=float(np.abs(controlled[0].astype(float)-controlled[1]).mean())
    assert strong_delta>5,f'Guidance has little measurable influence: {strong_delta}'
    times=[]
    for i in range(36):
        start=time.perf_counter();image=engine.process_frame(guides[i%3]);times.append(time.perf_counter()-start)
        assert image.std()>5 and np.isfinite(image).all()
    engine.close()
    report={'model':'dreamshaper-sketch','controlnet':'IDKiro/sdxs-512-dreamshaper-sketch',
            'frames':len(times),'mean_inference_ms':float(np.mean(times)*1000),
            'median_inference_ms':float(np.median(times)*1000),'p95_inference_ms':float(np.percentile(times,95)*1000),
            'inference_fps':float(1/np.mean(times)),'disabled_control_moved_input_pixel_mae':zero_delta,
            'enabled_control_moved_input_pixel_mae':strong_delta,'shape_strength':.8,
            'frame_storage':'RAM only; preview served from localhost memory if requested',
            'scope':'Inference including contour preprocessing; excludes TD transport and startup'}
    (ROOT/'artifacts/benchmark-sketch.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if preview_port is None:return
    rows=[np.concatenate([guides[i],no_control[i],controlled[i]],axis=1) for i in range(3)]
    preview=Image.fromarray(np.concatenate(rows,axis=0))
    stream=io.BytesIO();preview.save(stream,format='PNG');payload=stream.getvalue()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    print(f'RAM preview http://127.0.0.1:{preview_port}/preview.png ; columns input / control0 / control0.8',flush=True)
    HTTPServer(('127.0.0.1',preview_port),Handler).serve_forever()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview-port',type=int)
    main(parser.parse_args().preview_port)
