"""Terminal chat with the sales assistant.   python -m src.main [-v]"""
import logging
import sys

from src import llm
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

    print("Sales assistant ready (type 'exit' to quit).")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in {"exit", "quit"}:
            break
        if not msg:
            continue
        start = len(agent.trace)
        try:
            reply = agent.ask(msg)
        except Exception as e:  # keep the chat alive; the cause is in the log
            logging.getLogger("src").exception("turn failed")
            print("error>", e)
            continue
        if verbose:
            for t in agent.trace[start:]:
                if t["type"] == "tool":
                    print(f"  [tool] {t['name']}({t['args']}) -> {t['status']}")
            print("  [usage this message]", agent.last_usage.summary())
        print("bot>", reply)
        print("  steps:", describe_steps(agent.trace[start:]), "\n")
    print("Session usage:", llm.USAGE.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
