import main
from main import resolve_trace_dir


def test_no_trace_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DESK_PILOT_DIR", tmp_path)
    assert resolve_trace_dir(no_trace=True, keep_traces=False) is None


def test_default_creates_last_run(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DESK_PILOT_DIR", tmp_path)
    d = resolve_trace_dir(no_trace=False, keep_traces=False)
    assert d == tmp_path / "last_run"
    assert d.is_dir()


def test_default_overwrites_previous_last_run(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DESK_PILOT_DIR", tmp_path)
    first = resolve_trace_dir(no_trace=False, keep_traces=False)
    (first / "marker.txt").write_text("from previous run")
    second = resolve_trace_dir(no_trace=False, keep_traces=False)
    assert second == first
    assert not (second / "marker.txt").exists()


def test_keep_traces_creates_timestamped_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DESK_PILOT_DIR", tmp_path)
    d = resolve_trace_dir(no_trace=False, keep_traces=True)
    assert d.parent == tmp_path / "runs"
    assert d.is_dir()
