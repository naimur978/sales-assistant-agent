"""End-to-end agent evaluation: trajectory (which tools, what arguments, what status) and reply content.
Each case runs REPEATS times because model output varies between runs.
  python -m scripts.eval_agent [repeats] [case_id ...]"""
import json
import math
import re
import statistics
import sys
import time

from src.agent import Agent
from src.config import CACHE_DIR, ROOT

args = [a for a in sys.argv[1:]]
repeats = int(args.pop(0)) if args and args[0].isdigit() else 3
cases = json.loads((ROOT / "tests" / "agent_eval.json").read_text())
if args:
    cases = [c for c in cases if c["id"] in args]


def run_case(case):
    agent = Agent(log=False)
    reply, start = "", 0
    for turn in case["turns"]:
        start = len(agent.trace)
        reply = agent.ask(turn)
    tools = [t for t in agent.trace[start:] if t["type"] == "tool"]
    return agent, reply, tools


def check(case, reply, tools, trace):
    fails, low = [], reply.lower()
    called = [t["name"] for t in tools]
    for t in case.get("expect_tools", []):
        if t not in called:
            fails.append(f"expected tool {t}, called {called}")
    for t in case.get("forbid_tools", []):
        if t in called:
            fails.append(f"forbidden tool {t} was called")
    for name, want in case.get("expect_status", {}).items():
        if not any(t["name"] == name and t["status"] == want for t in tools):
            fails.append(f"{name} never returned status {want}")
    for name, bad in case.get("forbid_tools_status", {}).items():
        if any(t["name"] == name and t["status"] in bad for t in tools):
            fails.append(f"{name} returned a forbidden status {bad}")
    for name, want in case.get("args_include", {}).items():
        calls = [t["args"] for t in tools if t["name"] == name]
        if not any(all(a.get(k) == v for k, v in want.items()) for a in calls):
            fails.append(f"{name} args {calls} do not include {want}")
    for group in case.get("reply_has_any", []):
        if not any(g.lower() in low for g in group):
            fails.append(f"reply lacks any of {group}")
    for bad in case.get("reply_has_not", []):
        if bad.lower() in low:
            fails.append(f"reply contains forbidden text {bad!r}")
    for bad in case.get("reply_not_affirmed", []):
        for m in re.finditer(re.escape(bad.lower()), low):
            before = low[max(0, m.start() - 70):m.start()]
            if not re.search(r"\b(cannot|can't|not|no|never|without|nor)\b|n't", before):
                fails.append(f"reply asserts {bad!r}")
                break
    if case.get("asks_question") and "?" not in reply:
        fails.append("reply does not ask a question")
    guard = [t for t in trace if t["type"] == "guard"]
    return fails, guard


results, started = [], time.time()
for case in cases:
    runs = []
    for _ in range(repeats):
        t0 = time.time()
        agent, reply, tools = run_case(case)
        seconds = time.time() - t0
        fails, guard = check(case, reply, tools, agent.trace)
        runs.append({"seconds": seconds, "fails": fails, "guard": [g["problems"] for g in guard], "reply": reply,
                     "tools": [(t["name"], t["status"]) for t in tools]})
    ok = sum(not r["fails"] for r in runs)
    results.append({"id": case["id"], "passed": ok, "runs": runs})
    print(f"{ok}/{repeats}  {case['id']:<26} tools={runs[0]['tools']}" + ("" if ok == repeats else "   <-- FAIL"))

CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "agent_eval.json").write_text(json.dumps(results, indent=1, default=str))
total = sum(len(r["runs"]) for r in results)
passed = sum(r["passed"] for r in results)
guards = sum(bool(run["guard"]) for r in results for run in r["runs"])
print(f"\n{passed}/{total} runs passed | cases passing every repeat: {sum(r['passed'] == repeats for r in results)}/{len(results)} "
      f"| runs where the reply guard fired: {guards} | {time.time() - started:.0f}s")
times = sorted(run["seconds"] for r in results for run in r["runs"])
p95 = times[min(len(times) - 1, math.ceil(0.95 * len(times)) - 1)]
print(f"latency per case: median {statistics.median(times):.1f}s | p95 {p95:.1f}s | max {times[-1]:.1f}s ({len(times)} runs)")
for r in results:
    for i, run in enumerate(r["runs"]):
        if run["fails"]:
            print(f"\n[{r['id']} run {i}] {run['fails']}\n  tools={run['tools']}\n  reply: {run['reply'][:400]!r}")
