"""Measure inference in memory; persist numerical metrics only."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from backend.pipeline import MacDiffusion
from backend.model_specs import MODEL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)

def sample_frame(t=0):
    image = Image.new('RGB', (512,512), (14,18,35))
    draw = ImageDraw.Draw(image)
    for i in range(12):
        size = 230-i*16
        cx = 256+int(25*np.sin(t+i*0.2))
        cy = 256+int(25*np.cos(t+i*0.2))
        draw.ellipse((cx-size,cy-size,cx+size,cy+size), outline=(20+i*18,210-i*10,230-i*8), width=8)
    draw.polygon([(256,70),(420,360),(92,360)], fill=(255,125,35))
    draw.ellipse((195,205,317,327), fill=(5,20,45))
    return np.asarray(image)

def main(model='sdxs', frames=36):
    prompt = 'psychedelic sculpture, glowing neon rings, orange crystal pyramid, intricate detail, dark background'
    start=time.perf_counter()
    engine=MacDiffusion(prompt=prompt,model=model)
    loading=time.perf_counter()-start
    durations=[]
    for i in range(frames):
        inp=sample_frame(i*0.04)
        start=time.perf_counter()
        output=engine.process_frame(inp)
        durations.append(time.perf_counter()-start)
        assert output.shape==(512,512,3) and output.dtype==np.uint8
        assert output.std()>5, 'Generation is near-constant'
    stats={'model':MODEL_CONFIGS[model]['model_id'],'resolution':[512,512], 'frames':len(durations),
           'startup_seconds':loading,'mean_inference_ms':float(np.mean(durations)*1000),
           'median_inference_ms':float(np.median(durations)*1000),
           'p95_inference_ms':float(np.percentile(durations,95)*1000),
           'inference_fps':float(1/np.mean(durations)), 'prompt':prompt,
           'scope':'Model processing only; no TouchDesigner transport; excludes startup and warmup',
           'frame_storage':'RAM only; each frame is replaced, no images or videos saved'}
    filename = 'benchmark.json' if model == 'sdxs' else f'benchmark-{model}.json'
    (OUT/filename).write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=('sdxs','dreamshaper'),default='sdxs')
    parser.add_argument('--frames',type=int,default=36)
    args=parser.parse_args()
    main(model=args.model,frames=args.frames)
