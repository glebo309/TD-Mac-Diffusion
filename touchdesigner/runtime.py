import sys, pathlib, time, traceback
import numpy as np
import os
_install_file = pathlib.Path.home() / '.config/mac-diffusion/project.txt'
_registered = _install_file.read_text().strip() if _install_file.is_file() else ''
_source = pathlib.Path(__file__).resolve().parents[1] if '__file__' in globals() else None
_source = _source if _source and (_source / 'touchdesigner/bridge.py').is_file() else None
PROJECT = pathlib.Path(os.environ.get('MAC_DIFFUSION_HOME') or _registered or _source or pathlib.Path.home() / 'Projects/TD-Mac-Diffusion').expanduser()
if str(PROJECT/'touchdesigner') not in sys.path:sys.path.insert(0,str(PROJECT/'touchdesigner'))
from bridge import Bridge
from controls import edited_controls,prompt_seed,fresh_seed
last_prompt=None
client=None
output_array=None
last_capture=0
last_status=0
last_error=''
last_result_id=-1
previous_ui=None

def start():
    global client,previous_ui
    if client:client.close()
    previous_ui=None
    client=Bridge()

def sync_panel(c):
    global previous_ui
    current={'Running':bool(c.op('run').par.value0),
             'Prompt':str(c.op('prompt').panel.field.val),
             'Seed':int(c.op('seed').panel.field.val or 42),
             'Feedback':float(c.op('feedback').par.value0)*.95,
             'Wetdry':float(c.op('wetdry').par.value0)}
    for name,value in edited_controls(current,previous_ui).items():
        setattr(c.par,name,value)
    # Parameter expressions, CHOP exports and bindings remain authoritative.
    values={name:getattr(c.par,name).eval() for name in current}
    if current['Running']!=values['Running']:c.op('run').par.value0=values['Running']
    if current['Prompt']!=values['Prompt']:
        c.op('prompt/string')[0,0]=values['Prompt']
        c.op('prompt').panel.field.val=values['Prompt']
    if current['Seed']!=values['Seed']:
        c.op('seed/string')[0,0]=str(values['Seed'])
        c.op('seed').panel.field.val=str(values['Seed'])
    if abs(current['Feedback']-values['Feedback'])>1e-8:c.op('feedback').par.value0=values['Feedback']/.95
    if current['Wetdry']!=values['Wetdry']:c.op('wetdry').par.value0=values['Wetdry']
    previous_ui={'Running':bool(c.op('run').par.value0),
                 'Prompt':str(c.op('prompt').panel.field.val),
                 'Seed':int(c.op('seed').panel.field.val or 42),
                 'Feedback':float(c.op('feedback').par.value0)*.95,
                 'Wetdry':float(c.op('wetdry').par.value0)}

def set_text(node,text):
    if node.par.text.eval()!=text:node.par.text=text

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
    global last_capture,last_result_id,last_error,last_status,output_array
    try:
        if client is None:start()
        c=parent()
        sync_panel(c)
        update_prompt_seed(c)
        running=bool(c.par.Running)
        prompt=str(c.par.Prompt)
        controls={'running':running,'prompt':prompt,'seed':int(c.par.Seed),'feedback':float(c.par.Feedback),'model':str(c.par.Model),'shape_strength':float(c.par.Shapestrength)}
        now=time.monotonic()
        if now-last_capture>=1/20:
            data=c.op('input_image').numpyArray(delayed=True)
            if data is not None:
                # TOP numpy arrays have bottom-to-top row order. HTTP uses top-to-bottom RGB.
                rgb=np.ascontiguousarray(np.clip(data[::-1,:,:3]*255,0,255).astype(np.uint8))
                client.submit(rgb.tobytes(),controls)
            last_capture=now
        result,ident,status=client.snapshot()
        if result is not None and ident!=last_result_id:
            output_array=np.ascontiguousarray(result[::-1,:,:])
            c.op('generated').cook(force=True)
            last_result_id=ident
        if now-last_status<.25:return
        last_status=now
        fps=float(status.get('fps') or 0)
        latency=float(status.get('latency_ms') or 0)
        state=str(status.get('state','Connecting'))
        err=status.get('error') or ''
        model_label={'sdxs':'SDXS 0.9','dreamshaper':'SDXS DreamShaper','dreamshaper-sketch':'DreamShaper Shape'}[controls['model']]
        set_text(c.op('status_text'),('PAUSED' if not running else state.upper())+'    |    %.1f inference FPS    |    %.0f ms    |    %s / Core ML / 512 x 512'%(fps,latency,model_label))
        set_text(c.op('error_text'),err[:170])
        label='PAUSE' if running else 'START'
        if c.op('run').par.label.eval()!=label:c.op('run').par.label=label
        set_text(c.op('feedback_label_text'),'FEEDBACK   %.2f'%float(c.par.Feedback))
        set_text(c.op('mix_label_text'),'WET / DRY   %.0f%%'%(float(c.par.Wetdry)*100))
        set_text(c.op('seed_label_text'),'SEED   %d'%int(c.par.Seed))
        set_text(c.op('source_label_text'),'INPUT / CONNECTED TOP' if len(c.inputs) else 'INPUT / ANIMATED DEMO')
    except Exception:
        error=traceback.format_exc()
        if error!=last_error:
            (PROJECT/'touchdesigner/runtime_error.txt').write_text(error)
            last_error=error

def stop():
    global client
    if client:client.close();client=None
