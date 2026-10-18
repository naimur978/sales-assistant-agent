"""Checkpoint/resume for the service-visits table: a failure part-way loses nothing."""
import json

import pytest

from src import llm
from src.ingest import services
from src.ingest.services import Visit


@pytest.fixture
def setup(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    for i in (1, 2, 3):
        (reports / f"FSR-000{i}.txt").write_text(f"Report {i}. Total on-site hours: {i}.0 Total cost: ${i * 1000}")
    monkeypatch.setattr(services, "REPORT_DIR", reports)
    monkeypatch.setattr(services, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(services, "TABLE", tmp_path / "t.json")
    monkeypatch.setattr(services, "PARTIAL", tmp_path / "t.partial.jsonl")
    calls = []

    def fake_generate(contents, system, schema, purpose="extract"):
        n = int(contents.split("Report ")[1][0])
        calls.append(n)
        if n == 3 and fail["on"]:
            raise llm.QuotaExhausted("used up")
        return Visit(equipment="columncell", job_tags=["cleaning"], onsite_hours=float(n), total_cost=n * 1000.0, currency="USD")
    fail = {"on": True}
    monkeypatch.setattr("src.grounded._generate", fake_generate)
    return tmp_path, calls, fail


def test_failure_part_way_keeps_progress_and_rerun_resumes(setup):
    tmp, calls, fail = setup
    with pytest.raises(llm.QuotaExhausted):
        services.extract_all()
    saved = [json.loads(line)["report_id"] for line in (tmp / "t.partial.jsonl").read_text().splitlines()]
    assert saved == ["FSR-0001", "FSR-0002"] and not (tmp / "t.json").exists()

    fail["on"] = False
    calls.clear()
    rows = services.extract_all()
    assert calls == [3]  # only the missing report was extracted again
    assert [r["report_id"] for r in rows] == ["FSR-0001", "FSR-0002", "FSR-0003"]
    assert (tmp / "t.json").exists() and not (tmp / "t.partial.jsonl").exists()


def test_build_if_missing_reports_failure_instead_of_crashing(setup):
    assert services.build_if_missing() is False
    setup[2]["on"] = False
    assert services.build_if_missing() is True
    assert services.build_if_missing() is True  # already built: no more model calls
