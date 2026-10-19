"""Tool-calling agent loop with a step limit, validated tool arguments and a step trace."""
import json
import time

from google.genai import types
from pydantic import ValidationError

from src import llm, prompts
from src.config import HISTORY_TOKEN_BUDGET, MAX_INPUT_CHARS, TURN_DEADLINE_S
from src.guard import OFF_TOPIC, check_reply, looks_like_injection
from src.inputs import clean_input
from src.llm import Usage
from src.prompts import AGENT_SYSTEM
from src.slots import ungrounded
from src.tools.registry import Tool, build_tools
from src.trace import log_turn

MAX_STEPS = 6  # model calls per user message
STATUS_BY_TOOL = {"product_info": "searching the brochures and checking the quotes",
                  "estimate_service_cost": "building the estimate from past visits",
                  "assess_flotation": "analysing the plant data"}
SLOT_NAMES = ("equipment", "job", "country", "quantity", "silica", "iron", "silica_max", "iron_min")


def approx_tokens(content) -> int:
    """Rough size of one message (about 4 characters per token). Only used to decide when the chat is too long."""
    n = 0
    for p in content.parts or []:
        if p.text:
            n += len(p.text)
        if p.function_call:
            n += len(json.dumps(dict(p.function_call.args or {}), default=str))
        if p.function_response:
            n += len(json.dumps(p.function_response.response, default=str))
    return n // 4


def _llm(history, tools: list[Tool]):
    decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                       parameters_json_schema=t.args.model_json_schema()) for t in tools]
    cfg = types.GenerateContentConfig(
        temperature=0, system_instruction=AGENT_SYSTEM, tools=[types.Tool(function_declarations=decls)] if decls else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    return llm.generate(history, cfg, "agent")


class Agent:
    def __init__(self, tools: list[Tool] | None = None, llm=_llm, log: bool = True):
        self.context: dict = {}  # shared with the tools: the user's latest words, so a rewrite cannot lose key terms
        self.tools = {t.name: t for t in (tools if tools is not None else build_tools(self.context))}
        self.llm = llm
        self.history: list[types.Content] = []
        self.turn_starts: list[int] = []  # index in history where each user message begins
        self.known: dict = {}  # details the user gave that the code has checked; never dropped from the chat
        self.trace: list[dict] = []
        self.on_step = lambda step: None  # the terminal sets this to print each tool call as it finishes (-v)
        self.on_status = lambda text: None  # the terminal sets this to show what the assistant is doing
        self.tool_data: list[str] = []  # exact fields from tools; the reply guard accepts figures found here, but users never see it
        self.tool_texts: list[str] = []  # every tool result so far: the only allowed source of facts
        self.log = log
        self.last_usage = Usage()
        self._tool_start = 0
        self.user_text = ""  # everything the user has said so far; slot values must be traceable to it

    def _resolve(self, name: str) -> Tool | None:
        """The model sometimes decorates tool names (' name', 'default_api:name', odd prefix characters).
        Accept a name only if exactly one known tool name is a suffix of it, after dropping a namespace prefix."""
        bare = name.strip().split(":")[-1].split(".")[-1]
        hits = [t for n, t in self.tools.items() if bare == n or bare.endswith(n)]
        return hits[0] if len(hits) == 1 else None

    def _run_tool(self, name: str, raw_args: dict) -> dict:
        tool = self._resolve(name)
        if tool is None:
            return {"status": "error", "text": f"Unknown tool. Valid tool names: {', '.join(self.tools)}. "
                                               f"Call one of them, or answer the user without a tool."}
        try:
            args = tool.args(**raw_args)
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            return {"status": "invalid_arguments", "text": f"Arguments rejected ({msg}). Ask the user for correct values."}
        dump = args.model_dump()
        guessed = ungrounded(dump, self.user_text)
        for slot in SLOT_NAMES:  # remember every checked detail, even if the call still has to ask for another one
            if dump.get(slot) is not None and slot not in guessed:
                self.known[slot] = dump[slot]
        if guessed:
            hint = " For Larox equipment, ask whether it is the PF pressure filter or the PF-DS tower press." if "equipment" in guessed else ""
            return {"status": "needs_info", "text": f"The user has not said: {', '.join(guessed)}. Ask the user for it; "
                                                    f"never guess or reuse a value they did not give.{hint}"}
        try:
            return tool.run(args)
        except llm.LLMUnavailable:
            raise
        except Exception as e:  # a failing tool must not crash the chat or be papered over with guesses
            self.trace.append({"type": "tool_exception", "name": name, "error": repr(e)})
            return {"status": "error", "text": f"The tool failed ({type(e).__name__}). Say you could not get this data."}

    def _converse(self, message: str) -> str:
        self.user_text += " " + message
        self.turn_starts.append(len(self.history))
        self.history.append(types.Content(role="user", parts=[types.Part(text=self._pin_note() + message)]))
        deadline = time.monotonic() + TURN_DEADLINE_S
        for step in range(1, MAX_STEPS + 1):
            if time.monotonic() > deadline:
                self.trace.append({"step": step, "type": "deadline"})
                return "That is taking too long, so I stopped. Please try again in a moment."
            self.on_status("thinking")
            resp = self.llm(self.history, list(self.tools.values()))
            content = resp.candidates[0].content
            self.history.append(content)
            calls = [p.function_call for p in (content.parts or []) if p.function_call]
            if not calls:
                text = "".join(p.text or "" for p in content.parts if p.text and not getattr(p, "thought", False)).strip()
                self.trace.append({"step": step, "type": "reply", "text": text})
                return self._vet(text)
            replies = []
            for c in calls:
                tool = self._resolve(c.name)
                self.on_status(STATUS_BY_TOOL.get(tool.name, f"using {tool.name}") if tool else "thinking")
                result = self._run_tool(c.name, dict(c.args or {}))
                self.trace.append({"step": step, "type": "tool", "name": tool.name if tool else "unknown tool", "args": dict(c.args or {}),
                                   "status": result["status"], "result": result["text"]})
                self.on_step(self.trace[-1])
                self.tool_texts.append(result["text"])
                self.tool_data.append(json.dumps(result.get("data", {}), default=str))
                replies.append(types.Part.from_function_response(name=c.name, response=result))
            self.history.append(types.Content(role="user", parts=replies))
        self.trace.append({"step": MAX_STEPS, "type": "step_limit"})
        return "I could not finish that within my step limit. Could you rephrase or split the question?"

    def _vet(self, text: str) -> str:
        """Show the reply only if it passes the guard. One repair attempt, then fall back to the tool output itself."""
        self.on_status("checking the reply")
        problems = check_reply(text, self.tool_texts + self.tool_data, self.user_text)
        if not problems:
            return text or "Sorry, I could not produce an answer."
        self.trace.append({"type": "guard", "problems": problems, "stage": "first_reply", "text": text})
        self.history.append(types.Content(role="user", parts=[types.Part(text=(
            "Your reply was rejected: " + "; ".join(problems) + ". Rewrite it using only facts, numbers and product "
            "names that appear in the tool results. Keep numbers exactly as the tool gave them."))]))
        resp = self.llm(self.history, [])  # no tools: just rewrite
        content = resp.candidates[0].content
        self.history.append(content)
        retry = "".join(p.text or "" for p in (content.parts or []) if p.text and not getattr(p, "thought", False)).strip()
        problems = check_reply(retry, self.tool_texts + self.tool_data, self.user_text) if retry else ["empty reply"]
        if not problems:
            return retry
        self.trace.append({"type": "guard", "problems": problems, "stage": "after_repair"})
        if self.tool_texts:
            return "\n\n".join(
                self.tool_texts[self._tool_start:] or self.tool_texts[-1:])
        return OFF_TOPIC

    def _pin_note(self) -> str:
        """Checked details from earlier in the chat, put in front of the newest message so they are never forgotten."""
        if not self.known:
            return ""
        items = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in self.known.items())
        return f"[Already given by the user and checked by code: {items}]\n"

    def _trim_history(self) -> None:
        """Drop the oldest whole messages (with their tool calls) only when the chat is very long. Cutting at message
        boundaries means a tool result is never left without the call that produced it."""
        while len(self.turn_starts) > 1 and sum(approx_tokens(c) for c in self.history) > HISTORY_TOKEN_BUDGET:
            cut = self.turn_starts[1]
            del self.history[:cut]
            self.turn_starts = [i - cut for i in self.turn_starts[1:]]

    def ask(self, message: str) -> str:
        message = clean_input(message)
        if not message:
            return "Please type a question."
        if len(message) > MAX_INPUT_CHARS:
            return (f"That message is too long ({len(message):,} characters, the limit is {MAX_INPUT_CHARS:,}). "
                    f"Please shorten it or split it into parts.")
        if looks_like_injection(message):
            self.trace.append({"type": "guard", "problems": ["prompt injection"], "stage": "input"})
            return OFF_TOPIC
        self._trim_history()
        self.context["last_user_message"] = message
        start, self._tool_start = len(self.trace), len(self.tool_texts)
        before = llm.USAGE.snapshot()
        try:
            reply = self._converse(message)
        except llm.QuotaExhausted:
            reply = "The AI model's usage quota is used up for now, so I can't answer. Please try again later."
            self.trace.append({"type": "llm_error", "kind": "quota"})
        except llm.LLMUnavailable:
            reply = "I could not reach the AI model just now. Please try again in a moment."
            self.trace.append({"type": "llm_error", "kind": "unavailable"})
        self.last_usage = llm.USAGE.since(before)
        if self.log:
            log_turn(message, self.trace[start:], reply, self.last_usage, prompts.labels())
        return reply
