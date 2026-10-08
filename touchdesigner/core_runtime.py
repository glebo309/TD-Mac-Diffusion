"""Runtime for the network component. All TouchDesigner calls stay on its main thread."""
import pathlib
import sys
import time
import traceback
import numpy as np
import os
_install_file = pathlib.Path.home() / '.config/mac-diffusion/project.txt'
_registered = _install_file.read_text().strip() if _install_file.is_file() else ''
_source = pathlib.Path(__file__).resolve().parents[1] if '__file__' in globals() else None
_source = _source if _source and (_source / 'touchdesigner/bridge.py').is_file() else None
PROJECT = pathlib.Path(os.environ.get('MAC_DIFFUSION_HOME') or _registered or _source or pathlib.Path.home() / 'Projects/TD-Mac-Diffusion').expanduser()
if str(PROJECT/'touchdesigner') not in sys.path:sys.path.insert(0,str(PROJECT/'touchdesigner'))
from bridge import Bridge
from controls import prompt_seed,fresh_seed
last_prompt=None
client=None
output_array=None
last_capture=0
last_result_id=-1
last_error=''
last_controls=None

def start():
    # Loading a disabled component must not contact the shared worker.
    global client,last_capture,last_result_id,last_controls
    if client:client.close()
    client=None
    last_capture=0
    last_result_id=-1
    last_controls=None

def update_prompt_seed(c):
    global last_prompt
    prompt=str(c.par.Prompt)
    mode=str(getattr(c.par.Seed,'mode','CONSTANT')).split('.')[-1].upper()
    seed=prompt_seed(prompt,last_prompt,int(c.par.Seed),bool(c.par.Lockseed),mode!='CONSTANT')
    last_prompt=prompt
    if seed is not None:c.par.Seed=seed

def new_variation():
    global last_prompt
    c=parent()
    c.par.Seed=fresh_seed(int(c.par.Seed))
    last_prompt=str(c.par.Prompt)

def tick():
    global client,last_capture,last_result_id,last_error,last_controls,output_array
    try:
        c=parent()
        update_prompt_seed(c)
        controls={'running':bool(c.par.Running),'prompt':str(c.par.Prompt),
                  'seed':int(c.par.Seed),'feedback':float(c.par.Feedback),'model':str(c.par.Model),'shape_strength':float(c.par.Shapestrength)}
        if not controls['running']:
            if client is not None:client.submit(None,controls)
            c.store('worker_status',{'state':'bypassed'})
            return
        if client is None:client=Bridge()
        frame=None
        now=time.monotonic()
        if now-last_capture>=1/20:
            data=c.op('input_image').numpyArray(delayed=True)
            if data is not None:
                frame=np.ascontiguousarray(np.clip(data[::-1,:,:3]*255,0,255).astype(np.uint8)).tobytes()
            last_capture=now
        if frame is not None or controls!=last_controls:
            client.submit(frame,controls)
            last_controls=controls
        result,ident,status=client.snapshot()
        c.store('worker_status',status)
        if result is not None and ident!=last_result_id:
            output_array=np.ascontiguousarray(result[::-1,:,:])
            c.op('generated').cook(force=True)
            last_result_id=ident
    except Exception:
        error=traceback.format_exc()
        if error!=last_error:
            (PROJECT/'touchdesigner/core_runtime_error.txt').write_text(error)
            last_error=error

def stop():
    global client
    if client:client.close();client=None
