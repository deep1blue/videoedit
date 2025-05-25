import importlib.util
import sys
import types
from pathlib import Path
import subprocess

import pytest


def load_app(monkeypatch, tmp_path):
    # stub gradio module
    gradio = types.ModuleType("gradio")

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def upload(self, *args, **kwargs):
            return self

        def click(self, *args, **kwargs):
            return self

    class Blocks(Dummy):
        pass

    gradio.Blocks = Blocks
    gradio.Markdown = Dummy
    gradio.Row = Dummy
    gradio.Column = Dummy
    gradio.Video = Dummy
    gradio.Button = Dummy
    gradio.Textbox = Dummy
    gradio.Error = Exception
    monkeypatch.setitem(sys.modules, "gradio", gradio)

    # stub processing modules
    vu = types.ModuleType("processing.video_utils")
    vu.stabilize = lambda *a, **k: None
    vu.focus_speaker = lambda *a, **k: None
    vu.blur_bg = lambda *a, **k: None
    vu.adjust_levels = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "processing.video_utils", vu)

    au = types.ModuleType("processing.audio_utils")
    au.denoise = lambda *a, **k: None
    au.normalize_gain = lambda *a, **k: None
    au.cut_fillers = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "processing.audio_utils", au)

    tr = types.ModuleType("processing.transcript")
    tr.get_transcript = lambda *a, **k: []
    tr.find_filler_words = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "processing.transcript", tr)

    llm = types.ModuleType("processing.llm")
    llm.ask_video_question = lambda *a, **k: ""
    monkeypatch.setitem(sys.modules, "processing.llm", llm)

    # load module from path
    app_path = Path(__file__).resolve().parents[1] / "video-fixer" / "app.py"
    spec = importlib.util.spec_from_file_location("app", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    # redirect temp directory
    app.TMP_DIR = tmp_path
    tmp_path.mkdir(exist_ok=True)
    app.state = app.VideoState()
    return app


def test_upload(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    src = tmp_path / "src.mp4"
    src.write_text("data")
    dest = app.upload(src)
    assert dest.exists()
    assert dest.read_text() == "data"
    assert app.state.path == dest


def test_apply_video_effect(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    inp = tmp_path / "in.mp4"
    inp.write_text("in")
    app.state.path = inp

    called = {}

    def func(i, o):
        called["input"] = i
        called["output"] = o
        o.write_text("out")

    result = app.apply_video_effect(func)
    assert called["input"] == inp
    assert result == called["output"]
    assert result.exists()
    assert app.state.path == result


def test_apply_video_effect_no_video(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    with pytest.raises(app.gr.Error):
        app.apply_video_effect(lambda i, o: None)


def test_do_stabilize(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    called = {}
    monkeypatch.setattr(app, "apply_video_effect", lambda f: called.setdefault("func", f))
    app.do_stabilize()
    assert called["func"] is app.stabilize


def test_do_focus(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    called = {}
    monkeypatch.setattr(app, "apply_video_effect", lambda f: called.setdefault("func", f))
    app.do_focus()
    assert called["func"] is app.focus_speaker


def test_do_blur(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    called = {}
    monkeypatch.setattr(app, "apply_video_effect", lambda f: called.setdefault("func", f))
    app.do_blur()
    assert called["func"] is app.blur_bg


def test_do_contrast(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    called = {}
    monkeypatch.setattr(app, "apply_video_effect", lambda f: called.setdefault("func", f))
    app.do_contrast()
    assert called["func"] is app.adjust_levels


def test_do_denoise(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    called = {}
    monkeypatch.setattr(app, "apply_video_effect", lambda f: called.setdefault("func", f))
    app.do_denoise()
    assert called["func"] is app.denoise


def test_do_normalize(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    called = {}
    monkeypatch.setattr(app, "apply_video_effect", lambda f: called.setdefault("func", f))
    app.do_normalize()
    assert called["func"] is app.normalize_gain


def test_do_remove_fillers(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    inp = tmp_path / "in.mp4"
    inp.write_text("vid")
    app.state.path = inp

    audio = tmp_path / "a.wav"
    output = tmp_path / "cut.mp4"
    paths = [audio, output]

    def fake_temp_path(suffix=".mp4"):
        return paths.pop(0)

    monkeypatch.setattr(app, "_new_temp_path", fake_temp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(app, "get_transcript", lambda p: [(0.0, 1.0, "um")])
    monkeypatch.setattr(app, "find_filler_words", lambda t: [(0.0, 1.0)])

    called = {}

    def fake_cut(inp_v, out_v, ranges):
        called["inp"] = inp_v
        called["out"] = out_v
        called["ranges"] = ranges
        out_v.write_text("done")

    monkeypatch.setattr(app, "cut_fillers", fake_cut)

    res = app.do_remove_fillers()
    assert res == output
    assert app.state.path == output
    assert output.read_text() == "done"
    assert called["ranges"] == [(0.0, 1.0)]


def test_do_ask(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    with pytest.raises(app.gr.Error):
        app.do_ask("hi")

    inp = tmp_path / "in.mp4"
    inp.write_text("vid")
    app.state.path = inp
    monkeypatch.setattr(app, "ask_video_question", lambda p, q: f"ans:{q}")
    assert app.do_ask("question") == "ans:question"


def test_cleanup(tmp_path, monkeypatch):
    app = load_app(monkeypatch, tmp_path)
    (app.TMP_DIR / "file.txt").write_text("data")
    app.state.path = Path("some")
    app.cleanup()
    assert app.state.path is None
    assert list(app.TMP_DIR.iterdir()) == []
