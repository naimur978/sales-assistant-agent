"""Terminal chat with the sales assistant.   python -m src.main [-v]"""
import logging
import sys
import time

from src import llm
from src.display import Spinner, format_footer, format_reply, format_tool_call, paint, prompt, rule, use_color
from src.ingest import services
from src.trace import describe_steps


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("src").setLevel(logging.INFO)
    verbose = "-v" in sys.argv

    if not services.build_if_missing():
        print("Note: the service-visits table could not be built (model unavailable), so cost estimates are off "
              "for now. Progress is saved; start again later to resume.\n")
    try:
        from src.agent import Agent
        agent = Agent()
    except llm.LLMUnavailable as e:
        print(f"Cannot start: {e} Product search needs embeddings for the brochures on the first run. Try again later.")
        return 1

    color = use_color()
    print(paint("Sales assistant", "bold", color) + "  (type 'exit' to quit" + (", -v shows tool calls and usage)" if verbose else ", run with -v to see tool calls)"))
    print(rule(color))
    while True:
        try:
            msg = input(prompt(color)).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in {"exit", "quit"}:
            break
        if not msg:
            continue
        start = len(agent.trace)
        began = time.perf_counter()
        try:
            with Spinner(color=color) as spinner:
                agent.on_status = spinner.update
                agent.on_step = (lambda t: spinner.log(format_tool_call(t, color)) if t["type"] == "tool" else None) if verbose else (lambda t: None)
                reply = agent.ask(msg)
        except Exception as e:  # keep the chat alive; the cause is in the log
            logging.getLogger("src").exception("turn failed")
            print("error>", e)
            continue
        steps = agent.trace[start:]
        print()
        print(paint("assistant", "green", color))
        print(format_reply(reply, color))
        print()
        print(format_footer(describe_steps(steps), agent.last_usage.short() if verbose else None, color,
                            seconds=time.perf_counter() - began))
        print(rule(color))
    print("Session usage:", llm.USAGE.short())
    return 0


if __name__ == "__main__":
    sys.exit(main())
