"""Non-blocking local frame transport. This module never calls TouchDesigner APIs."""
import json
import errno
import os
import pathlib
import subprocess
import threading
import time
import urllib.request
import urllib.error
import numpy as np

class Bridge:
    def __init__(self, url='http://127.0.0.1:8765'):
        self.url=url
        self.project=pathlib.Path(__file__).resolve().parents[1]
        self.worker_process=None
        self.last_launch=0
        self.lock=threading.Lock()
        self.frame=None
        self.result=None
        self.result_id=-1
        self.status={'state':'Connecting to local worker','fps':0,'latency_ms':0}
        self.controls={}
        self.sent_controls=None
        self.closed=False
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
    def submit(self, rgb, controls):
        with self.lock:
            self.frame=rgb
            self.controls=dict(controls)
    def snapshot(self):
        with self.lock:
            return self.result,self.result_id,dict(self.status)
    def close(self):
        self.closed=True
    def _request(self,path,data=None,json_data=False):
        headers={'Content-Type':'application/json' if json_data else 'application/octet-stream'}
        request=urllib.request.Request(self.url+path,data=data,headers=headers)
        return urllib.request.urlopen(request,timeout=3)
    def _maybe_launch_worker(self,exc):
        if getattr(self,'closed',False):return
        reason=exc.reason if isinstance(exc,urllib.error.URLError) else exc
        if not isinstance(reason,ConnectionRefusedError) and getattr(reason,'errno',None)!=errno.ECONNREFUSED:
            return
        if self.worker_process is not None and self.worker_process.poll() is None:
            return
        if time.monotonic()-self.last_launch<10:
            return
        python=self.project/'.venv/bin/python'
        if not python.exists():
            return
        self.last_launch=time.monotonic()
        env=os.environ.copy()
        env.pop('PYTHONHOME',None)
        env.pop('PYTHONPATH',None)
        with (self.project/'worker.log').open('ab') as log:
            self.worker_process=subprocess.Popen([str(python),'-m','worker.server'],cwd=str(self.project),stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)

    def _run(self):
        while not self.closed:
            try:
                with self.lock:
                    frame=self.frame; self.frame=None
                    controls=dict(self.controls)
                if controls and controls!=self.sent_controls:
                    with self._request('/control',json.dumps(controls).encode(),True) as response:
                        response.read()
                    self.sent_controls=controls
                if frame is not None and controls.get('running',False):
                    with self._request('/frame',frame) as response:response.read()
                with self._request('/result') as response:
                    if response.status==200:
                        result_id=int(response.headers.get('X-Frame-Id',0))
                        data=response.read()
                        if len(data)==512*512*3 and result_id!=self.result_id:
                            result=np.frombuffer(data,dtype=np.uint8).reshape(512,512,3).copy()
                            with self.lock:self.result=result;self.result_id=result_id
                with self._request('/status') as response:
                    status=json.loads(response.read())
                with self.lock:self.status=status
                time.sleep(.012)
            except Exception as exc:
                self.sent_controls=None
                self._maybe_launch_worker(exc)
                with self.lock:self.status={'state':'Worker unavailable','error':str(exc),'fps':0,'latency_ms':0}
                time.sleep(.5)
