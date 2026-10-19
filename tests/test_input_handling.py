"""Input handling: cleaning, limits, history trimming, country and equipment grounding, readable replies. No network."""

from src.agent import Agent
from src.config import MAX_INPUT_CHARS
from src.inputs import clean_input
from src.slots import ungrounded
from src.tools.service_cost import EstimateRequest, estimate, render
from tests.test_agent import call, echo_tool, fake_llm, say
from tests.test_service_cost import VISITS


# ---- cleaning and limits
def test_control_characters_and_extra_spaces_are_removed():
    assert clean_input("  hello\x00 \x07world\t\tagain\n ok ") == "hello world again\n ok"


def test_empty_message_gets_a_prompt_not_a_model_call():
    calls = []
    a = Agent(tools=[echo_tool()], llm=lambda h, t: calls.append(1), log=False)
    assert "type a question" in a.ask("   \x00 ") and calls == []


def test_too_long_message_is_refused_before_any_model_call():
    calls = []
    a = Agent(tools=[echo_tool()], llm=lambda h, t: calls.append(1), log=False)
    reply = a.ask("x" * (MAX_INPUT_CHARS + 1))
    assert "too long" in reply and str(f"{MAX_INPUT_CHARS:,}") in reply and calls == []


# ---- history: nothing is dropped in a normal chat, and checked details are never forgotten
def cost_tool(runs):
    from src.tools.registry import CostArgs, Tool
    return Tool("cost", "cost", CostArgs, lambda a: runs.append(a) or {"status": "ok", "text": "estimate"})


def test_a_normal_long_chat_is_not_trimmed():
    script = [call("echo", text="x"), say("ok")] * 40
    a = Agent(tools=[echo_tool()], llm=fake_llm(script), log=False)
    for i in range(30):
        a.ask(f"message {i}")
    assert len(a.turn_starts) == 30 and a.turn_starts[0] == 0


def test_only_a_very_long_chat_is_trimmed_and_only_at_message_boundaries(monkeypatch):
    import src.agent as agent_mod
    monkeypatch.setattr(agent_mod, "HISTORY_TOKEN_BUDGET", 150)
    script = [call("echo", text="x" * 200), say("ok")] * 40
    a = Agent(tools=[echo_tool()], llm=fake_llm(script), log=False)
    for i in range(15):
        a.ask(f"m{i}")
    assert 1 <= len(a.turn_starts) < 15
    kinds = [("call" if any(p.function_call for p in c.parts) else "resp" if any(p.function_response for p in c.parts) else "text")
             for c in a.history]
    assert kinds[0] == "text" and a.turn_starts[0] == 0
    assert all(kinds[i - 1] == "call" for i, k in enumerate(kinds) if k == "resp")  # no orphaned tool result
    assert "m0" in a.user_text  # the slot checks still know everything the user said


def seen_by_model(history):
    return history[-1].parts[0].text


def test_checked_details_are_pinned_to_every_later_message():
    runs, sent = [], []
    script = iter([call("cost", equipment="larox_pf_ds", job="cloth_replacement", country="Chile"), say("How many cloths?"),
                   call("cost", equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), say("done")])

    def llm(history, tools):
        sent.append(seen_by_model(history) if history[-1].role == "user" and history[-1].parts[0].text else None)
        return SimpleNamespace(candidates=[SimpleNamespace(content=next(script))])

    from types import SimpleNamespace
    a = Agent(tools=[cost_tool(runs)], llm=llm, log=False)
    a.ask("Larox PF-DS cloth change in Chile")
    a.ask("12 cloths")
    second = next(t for t in sent[2:] if t)
    assert "equipment=larox_pf_ds" in second and "country=Chile" in second and "12 cloths" in second
    assert a.known["quantity"] == 12 and runs[-1].quantity == 12


def test_a_detail_the_user_never_said_is_not_pinned():
    runs = []
    a = Agent(tools=[cost_tool(runs)], llm=fake_llm([call("cost", equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=7),
                                                      say("How many?")]), log=False)
    a.ask("Larox PF-DS cloth change in Chile")  # no quantity given, the model invented 7
    assert "quantity" not in a.known and a.known["country"] == "Chile" and runs == []


def test_pinned_details_survive_trimming(monkeypatch):
    import src.agent as agent_mod
    monkeypatch.setattr(agent_mod, "HISTORY_TOKEN_BUDGET", 100)
    runs = []
    script = [call("cost", equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), say("ok")] + \
             [call("echo", text="y" * 300), say("ok")] * 20
    a = Agent(tools=[cost_tool(runs), echo_tool()], llm=fake_llm(script), log=False)
    a.ask("12 cloths on a Larox PF-DS in Chile")
    for i in range(8):
        a.ask(f"another message {i}")
    assert "12 cloths" not in " ".join(p.text or "" for c in a.history for p in c.parts)  # the original message is gone
    assert a._pin_note().startswith("[Already given") and "quantity=12" in a._pin_note()


# ---- countries
def test_country_aliases_accents_demonyms_and_typos():
    base = dict(equipment="larox_pf_ds", job="cloth_replacement", quantity=12)
    ok = lambda country, text: "country" not in ungrounded({**base, "country": country}, text)
    assert ok("Peru", "12 cloths on a Larox PF-DS in Perú")
    assert ok("Chile", "Chilean site, 12 cloths on the Larox PF-DS tower press")
    assert ok("Sweden", "Swedish mine, Larox PF-DS, 12 cloths")
    assert ok("United Kingdom", "12 cloths on a Larox PF-DS in the UK")
    assert ok("Chile", "12 cloths on a Larox PF-DS in Chille")  # typo
    assert not ok("Chile", "12 cloths on a Larox PF-DS in Peru")
    assert not ok("Chile", "12 cloths on a Larox PF-DS")
    assert not ok("United States", "12 cloths on a Larox PF-DS, it's urgent")  # 'us' must be a whole word


# ---- Larox PF vs PF-DS
def test_plain_larox_is_ambiguous_and_pf_ds_is_not_accepted_as_pf():
    job = dict(job="cloth_replacement", country="Chile", quantity=12)
    assert "equipment" in ungrounded({**job, "equipment": "larox_pf"}, "12 cloths on our Larox filter in Chile")
    assert "equipment" in ungrounded({**job, "equipment": "larox_pf_ds"}, "12 cloths on our Larox filter in Chile")
    assert "equipment" in ungrounded({**job, "equipment": "larox_pf"}, "12 cloths on a Larox PF-DS in Chile")
    assert ungrounded({**job, "equipment": "larox_pf"}, "12 cloths on a Larox PF press in Chile") == []
    assert ungrounded({**job, "equipment": "larox_pf_ds"}, "12 cloths on a Larox pfds in Chile") == []


# ---- readings written in different ways
def test_readings_with_comma_decimals_percent_words_and_labels():
    assert ungrounded(dict(silica=3.1, iron=64.5), "silica 3,1 % and iron 64,5 % in the concentrate") == []
    assert ungrounded(dict(silica=3.1, iron=64.5), "conc is 3.1 percent silica, 64.5 percent Fe") == []
    # swapped values are caught because each number must sit next to its own label
    assert set(ungrounded(dict(silica=64.5, iron=3.1), "3.1% silica and 64.5% iron")) == {"silica", "iron"}


# ---- readable reply
def test_estimate_heading_uses_plain_words_not_internal_names():
    text = render(estimate(EstimateRequest(equipment="larox_pf_ds", job="cloth_replacement", country="Chile", quantity=12), VISITS))
    first = text.splitlines()[0]
    assert first.startswith("Estimate for replacing filter cloths (12) on a Larox PF-DS tower press filter in Chile")
    assert "larox_pf_ds" not in first and "cloth_replacement" not in first


def test_canonical_country_maps_typed_names_onto_the_data_spelling():
    from src.slots import canonical_country
    known = ["Chile", "Peru", "South Africa", "Sweden"]
    assert canonical_country("chille", known) == "Chile" and canonical_country("Peruvian", known) == "Peru"
    assert canonical_country("swedish", known) == "Sweden" and canonical_country("south africa", known) == "South Africa"
    assert canonical_country("Norway", known) == "Norway" and canonical_country("UK", known) == "UK"
