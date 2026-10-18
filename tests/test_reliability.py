"""Retries, timeouts, quota handling, usage counting and prompt versioning. No network."""
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from src import config, llm, prompts
from src.agent import Agent
from tests.test_agent import call, echo_tool, fake_llm, say


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    waits = []
    monkeypatch.setattr(llm, "sleep", waits.append)
    monkeypatch.setattr(llm, "USAGE", llm.Usage())
    return waits


def api_error(code, text="x"):
    return errors.ClientError(code, {"error": {"code": code, "message": text, "status": "X"}})


def flaky(failures, then="ok"):
    seq = list(failures)

    def fn():
        if seq:
            raise seq.pop(0)
        return then
    return fn


def test_temporary_error_is_retried_then_succeeds(_fast):
    assert llm.call(flaky([api_error(503), api_error(429, "per minute")])) == "ok"
    assert len(_fast) == 2 and llm.USAGE.retries == 2 and llm.USAGE.calls == 1


def test_gives_up_after_max_attempts_with_a_clear_error(_fast):
    with pytest.raises(llm.LLMUnavailable):
        llm.call(flaky([api_error(503)] * 10))
    assert len(_fast) == config.MAX_ATTEMPTS - 1 and llm.USAGE.failures == 1


def test_daily_quota_fails_fast_without_waiting(_fast):
    with pytest.raises(llm.QuotaExhausted):
        llm.call(flaky([api_error(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")]))
    assert _fast == []  # no sleeping for hours


def test_client_errors_are_not_retried(_fast):
    with pytest.raises(llm.LLMUnavailable):
        llm.call(flaky([api_error(400, "bad request")]))
    assert _fast == []


def test_timeout_is_retried(_fast):
    assert llm.call(flaky([httpx.ReadTimeout("slow")])) == "ok" and len(_fast) == 1


def test_programming_errors_are_not_swallowed():
    with pytest.raises(ValueError):
        llm.call(flaky([ValueError("bug")]))


def test_backoff_grows(_fast):
    with pytest.raises(llm.LLMUnavailable):
        llm.call(flaky([api_error(503)] * 10))
    assert _fast[0] < _fast[1] < _fast[2] + 1


def test_token_usage_is_counted_per_purpose(monkeypatch):
    meta = SimpleNamespace(prompt_token_count=120, candidates_token_count=30)
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **k: SimpleNamespace(usage_metadata=meta))))
    llm.generate("hi", types.GenerateContentConfig(), "agent")
    llm.generate("hi", types.GenerateContentConfig(), "extract")
    assert (llm.USAGE.input_tokens, llm.USAGE.output_tokens, llm.USAGE.calls) == (240, 60, 2)
    assert llm.USAGE.by_purpose["agent"]["input_tokens"] == 120


def test_cost_is_not_invented_when_prices_are_unset(monkeypatch):
    u = llm.Usage(calls=1, input_tokens=1_000_000, output_tokens=500_000)
    assert u.cost_usd() is None and "set PRICE_PER_MILLION_TOKENS" in u.summary()
    monkeypatch.setattr(config, "PRICE_PER_MILLION_TOKENS", {"input": 0.10, "output": 0.40})
    assert u.cost_usd() == pytest.approx(0.30)


def test_agent_explains_quota_instead_of_crashing_or_misleading():
    def boom(history, tools):
        raise llm.QuotaExhausted("used up")
    a = Agent(tools=[echo_tool()], llm=boom, log=False)
    reply = a.ask("hello")
    assert "quota" in reply.lower() and a.trace[-1]["kind"] == "quota"


def test_quota_inside_a_tool_is_not_reported_as_not_in_brochures():
    def tool_run(a):
        raise llm.QuotaExhausted("used up")
    from pydantic import BaseModel

    from src.tools.registry import Tool

    class A(BaseModel):
        text: str
    a = Agent(tools=[Tool("echo", "echo", A, tool_run)], llm=fake_llm([call("echo", text="x"), say("never reached")]), log=False)
    assert "quota" in a.ask("go").lower()


def test_turn_deadline_stops_a_slow_turn(monkeypatch):
    import src.agent as agent_mod
    monkeypatch.setattr(agent_mod, "TURN_DEADLINE_S", -1)
    a = Agent(tools=[echo_tool()], llm=fake_llm([say("never")]), log=False)
    assert "taking too long" in a.ask("go") and a.trace[-1]["type"] == "deadline"


def test_log_records_usage_and_prompt_versions(tmp_path, monkeypatch):
    import json

    import src.trace as tr
    monkeypatch.setattr(tr, "LOG", tmp_path / "t.jsonl")
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path)
    a = Agent(tools=[echo_tool()], llm=fake_llm([say("hello")]), log=True)
    a.ask("hi")
    rec = json.loads((tmp_path / "t.jsonl").read_text())
    assert "usage" in rec and any(p.startswith("agent_system@") for p in rec["prompts"])


# ---- prompt files ----
def test_every_prompt_file_has_a_version_and_label():
    for f in prompts.PROMPT_DIR.glob("*.md"):
        text, label = prompts.load(f.stem)
        assert f.read_text().startswith("version:") and text and label.startswith(f"{f.stem}@")


def test_prompt_hash_changes_when_text_changes(tmp_path, monkeypatch):
    (tmp_path / "p.md").write_text("version: 1\n---\nhello")
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    prompts.load.cache_clear()
    first = prompts.load("p")[1]
    (tmp_path / "p.md").write_text("version: 1\n---\nhello world")
    prompts.load.cache_clear()
    assert prompts.load("p")[1] != first
    prompts.load.cache_clear()
