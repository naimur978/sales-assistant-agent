"""The agent-computer interface: tool names, descriptions and parameter documentation, checked without the API."""
import json
import re

from google.genai import types

from src.tools.registry import TOOL_SPECS
from src.trace import describe_steps


def test_tool_names_are_unique_snake_case():
    names = list(TOOL_SPECS)
    assert len(names) == len(set(names)) == 3
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", n) for n in names)


def test_each_description_is_informative_but_not_bloated():
    for name, (desc, _) in TOOL_SPECS.items():
        assert 150 <= len(desc) <= 900, name


def test_each_description_says_when_not_to_use_the_tool():
    for name, (desc, _) in TOOL_SPECS.items():
        assert "Not for" in desc, f"{name} does not say what it is not for"


def test_each_description_says_what_the_tool_needs_or_returns():
    for name, (desc, _) in TOOL_SPECS.items():
        assert re.search(r"Needs|Returns|Use for", desc), name


def test_every_parameter_is_documented():
    for name, (_, model) in TOOL_SPECS.items():
        for field, spec in model.model_json_schema()["properties"].items():
            assert spec.get("description"), f"{name}.{field} has no description"


def test_choice_parameters_list_their_allowed_values_and_numbers_have_limits():
    cost = TOOL_SPECS["estimate_service_cost"][1].model_json_schema()
    dumped = json.dumps(cost)
    for allowed in ("larox_pf_ds", "cloth_replacement", "emergency_repair"):
        assert allowed in dumped
    flotation = TOOL_SPECS["assess_flotation"][1].model_json_schema()["properties"]
    assert flotation["silica"]["maximum"] == 100 and flotation["iron"]["minimum"] == 0


def test_required_parameters_are_right():
    assert set(TOOL_SPECS["assess_flotation"][1].model_json_schema()["required"]) == {"silica", "iron"}
    assert set(TOOL_SPECS["product_info"][1].model_json_schema()["required"]) == {"question"}
    assert "required" not in TOOL_SPECS["estimate_service_cost"][1].model_json_schema()  # the tool asks for what is missing


def test_declarations_build_for_the_model():
    for name, (desc, model) in TOOL_SPECS.items():
        d = types.FunctionDeclaration(name=name, description=desc, parameters_json_schema=model.model_json_schema())
        assert d.name == name and d.parameters_json_schema["type"] == "object"


# ---- the steps line shown after every reply
def test_steps_line_lists_tools_and_statuses():
    steps = [{"type": "tool", "name": "estimate_service_cost", "status": "needs_info"},
             {"type": "tool", "name": "estimate_service_cost", "status": "estimate"}]
    assert describe_steps(steps) == "estimate_service_cost (needs_info) -> estimate_service_cost (estimate)"


def test_steps_line_says_so_when_no_tool_was_used_or_the_reply_was_rewritten():
    assert describe_steps([{"type": "reply"}]) == "answered directly, no tool used"
    assert "reply rewritten" in describe_steps([{"type": "tool", "name": "product_info", "status": "answered"},
                                                {"type": "guard", "stage": "first_reply"}])
