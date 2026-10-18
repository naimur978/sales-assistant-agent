import pytest


@pytest.fixture(autouse=True)
def _no_real_logs(tmp_path, monkeypatch):
    import src.trace as tr
    monkeypatch.setattr(tr, "LOG", tmp_path / "traces.jsonl")
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path)
