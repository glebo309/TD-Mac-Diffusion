import errno
import pathlib
import urllib.error
from unittest.mock import Mock
from touchdesigner.bridge import Bridge
from touchdesigner.controls import edited_controls

def test_unchanged_ui_does_not_override_automated_parameters():
    ui={'Prompt':'old prompt','Seed':42,'Running':True,'Feedback':.43,'Wetdry':1.0}
    assert edited_controls(ui,ui.copy())=={}

def test_only_manually_changed_control_is_written():
    previous={'Prompt':'old','Seed':42,'Running':True}
    assert edited_controls({'Prompt':'new','Seed':42,'Running':True},previous)=={'Prompt':'new'}

def test_initial_sync_does_not_overwrite_parameters():
    assert edited_controls({'Prompt':'stale UI'},None)=={}

def test_launch_worker_only_after_connection_refused(monkeypatch,tmp_path):
    bridge=Bridge.__new__(Bridge)
    bridge.project=tmp_path
    bridge.worker_process=None
    bridge.last_launch=0
    python=tmp_path/'.venv/bin/python';python.parent.mkdir(parents=True);python.touch()
    popen=Mock(return_value=Mock(poll=Mock(return_value=None)))
    monkeypatch.setattr('touchdesigner.bridge.subprocess.Popen',popen)
    bridge._maybe_launch_worker(TimeoutError('slow server'))
    popen.assert_not_called()
    refusal=urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED,'refused'))
    bridge.closed=True
    bridge._maybe_launch_worker(refusal)
    popen.assert_not_called()
    bridge.closed=False
    bridge._maybe_launch_worker(refusal)
    assert popen.call_count==1
    bridge._maybe_launch_worker(refusal)
    assert popen.call_count==1
    assert popen.call_args.args[0][-2:]==['-m','worker.server']

def test_float_round_trip_is_not_a_manual_edit():
    assert edited_controls({'Feedback':.43000000000000005},{'Feedback':.43})=={}

def test_prompt_variation_changes_once_and_holds_seed_between_frames():
    from touchdesigner.controls import prompt_seed
    choose=Mock(return_value=1234)
    assert prompt_seed('cat',None,42,False,False,choose) is None
    assert prompt_seed('cat','cat',42,False,False,choose) is None
    assert prompt_seed('cat','dog',42,False,False,choose)==1234
    assert prompt_seed('cat','cat',1234,False,False,choose) is None
    choose.assert_called_once()

def test_locked_or_automated_seed_is_preserved_on_prompt_change():
    from touchdesigner.controls import prompt_seed
    choose=Mock(return_value=1234)
    assert prompt_seed('cat','dog',42,True,False,choose) is None
    assert prompt_seed('cat','dog',42,False,True,choose) is None
    choose.assert_not_called()

def test_explicit_variation_never_repeats_current_seed():
    from touchdesigner.controls import fresh_seed
    assert fresh_seed(42,Mock(return_value=42))==43
    assert fresh_seed(42,Mock(return_value=25))==25

def test_new_bridge_closes_previous_one(monkeypatch):
    monkeypatch.setattr(Bridge,'_run',lambda self:None)
    first=Bridge()
    second=Bridge()
    assert first.closed and not second.closed

def test_smoothing_feedback_reads_the_blended_image():
    root=pathlib.Path(__file__).resolve().parents[1]/'touchdesigner/component/mac_diffusion_core'
    # The feedback input is only the reset image; wiring smooth into it makes a cook loop.
    assert '0 \tgenerated\n' in (root/'smooth_feedback.n').read_text()
    assert 'top 0 smooth' in (root/'smooth_feedback.parm').read_text()

def test_component_creation_closes_the_previous_bridge():
    root=pathlib.Path(__file__).resolve().parents[1]/'touchdesigner/component/mac_diffusion_core'
    start=Mock()
    runtime=Mock()
    runtime.module.start=start
    namespace={'op':lambda name:runtime}
    exec((root/'execute.text').read_bytes()[27:].decode(),namespace)
    namespace['onCreate']()
    start.assert_called_once_with()

def test_component_enables_its_creation_callback():
    root=pathlib.Path(__file__).resolve().parents[1]/'touchdesigner/component/mac_diffusion_core'
    assert 'create 0 on' in (root/'execute.parm').read_text().splitlines()

def test_closed_bridge_does_not_post_after_in_flight_request():
    import threading
    from unittest.mock import MagicMock
    bridge=Bridge.__new__(Bridge)
    bridge.lock=threading.Lock();bridge.closed=False;bridge.sent_controls=None
    bridge.frame=b'frame';bridge.controls={'running':True}
    paths=[]
    def request(path,data=None,json_data=False):
        paths.append(path)
        bridge.closed=True  # a reload closes this bridge while /control is in flight
        return MagicMock()
    bridge._request=request
    bridge._run()
    assert paths==['/control']

def test_worker_older_than_source_is_stopped_once(monkeypatch,tmp_path):
    bridge=Bridge.__new__(Bridge)
    bridge.project=tmp_path;bridge.checked_pid=None
    source=tmp_path/'worker/service.py';source.parent.mkdir();source.touch()
    edited=source.stat().st_mtime
    kill=Mock();monkeypatch.setattr('touchdesigner.bridge.os.kill',kill)
    ps=Mock(return_value=Mock(stdout='/x/.venv/bin/python -m worker.server'))
    monkeypatch.setattr('touchdesigner.bridge.subprocess.run',ps)
    bridge._stop_if_stale({'pid':11,'started':edited+5})
    kill.assert_not_called()
    bridge._stop_if_stale({'pid':12,'started':edited-5})
    kill.assert_called_once()
    assert kill.call_args.args[0]==12
    bridge._stop_if_stale({'pid':12,'started':edited-5})
    assert kill.call_count==1
    ps.return_value=Mock(stdout='/usr/bin/some-other-app')
    bridge._stop_if_stale({'pid':13,'started':edited-5})
    assert kill.call_count==1
    bridge._stop_if_stale({'started':edited-5})  # pre-pid worker: nothing to signal
    assert kill.call_count==1
