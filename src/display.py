"""How the chat looks in the terminal. Pure formatting, no logic about answers."""
import os
import re
import shutil
import sys
import textwrap
import threading

STYLES = {"dim": "2", "bold": "1", "cyan": "36", "green": "32", "yellow": "33"}
CITATION = re.compile(r"\[page \d+ of [a-z0-9-]+\.pdf\]")
GLUED = re.compile(r"\[page\u00a0\d+\u00a0of\u00a0[a-z0-9-]+\.pdf\]")
# The model sometimes rewrites a source as "[file, page 2]" or "[file, p.2]". Show one wording always.
CITATION_VARIANT = re.compile(r"\[([a-z0-9-]+), (?:page|p\.?) ?(\d+)\]")


def use_color(stream=None) -> bool:
    return (stream or sys.stdout).isatty() and "NO_COLOR" not in os.environ


def paint(text: str, style: str, on: bool) -> str:
    return f"\033[{STYLES[style]}m{text}\033[0m" if on else text


def line_width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 100)) - 4


def _plain_math(expr: str) -> str:
    for a, b in ((r"\geq", ">="), (r"\ge", ">="), (r"\leq", "<="), (r"\le", "<="), (r"\times", "x"), (r"\%", "%"),
                 (r"\,", " "), ("R^2", "R2"), ("^2", "2")):
        expr = expr.replace(a, b)
    return re.sub(r"[{}\\]", "", expr).strip()


def clean_markup(text: str) -> str:
    """A terminal does not render Markdown or LaTeX, so remove the symbols and keep the words."""
    text = re.sub(r"\$([^$\n]+)\$", lambda m: _plain_math(m.group(1)), text)
    text = re.sub(r"(?m)^(\s*)\*\s+", r"\1- ", text)  # "* item" bullets
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"\1", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    return re.sub(r"[ \t]*\[page \d+ of [a-z0-9_-]+\]", "", text)  # a page number with no PDF behind it is invented


def normalise_citations(text: str) -> str:
    return CITATION_VARIANT.sub(lambda m: f"[page {m.group(2)} of {m.group(1)}.pdf]", text)


# "Recommendation: X. You said "..."; the brochure says: ..." becomes a small labelled block
REC_BLOCK = re.compile(r'^Recommendation: (?P<product>[^.]+)\.\s*You said "(?P<said>[^"]+)";?\s*the brochure says:?\s*(?P<says>.*)$')
LABEL_WIDTH = 16


def _labelled(label: str, value: str, width: int, color: bool) -> str:
    pad = " " * (2 + LABEL_WIDTH)
    text = textwrap.fill(value, width, initial_indent="  " + label.ljust(LABEL_WIDTH), subsequent_indent=pad)
    return paint(text[: 2 + LABEL_WIDTH], "dim", color) + text[2 + LABEL_WIDTH:]


def _is_heading(body: str) -> bool:
    return body.endswith(":") and not body.startswith("-") and ". " not in body and len(body) <= 80


def _no_colons(text: str) -> str:
    """Terminal answers read better without colons: a trailing one goes, one before a space becomes a dash."""
    return re.sub(r":\s+", " - ", re.sub(r":\s*$", "", text))


def format_reply(text: str, color: bool = False) -> str:
    """Lay the answer out like a short briefing. A recommendation becomes a labelled block, section titles
    such as 'Other relevant facts' become bold headings, long lines wrap, sources and notes get their colours."""
    # glue each citation with non-breaking spaces so wrapping never splits it across two lines
    text = CITATION.sub(lambda m: m.group(0).replace(" ", "\u00a0"), normalise_citations(clean_markup(text)))
    width, out = line_width(), []
    for raw in text.split("\n"):
        if not raw.strip():
            out.append("")
            continue
        lead = len(raw) - len(raw.lstrip(" "))
        body = raw.strip()
        rec = REC_BLOCK.match(body)
        if rec:
            head = "  " + "Recommendation".ljust(LABEL_WIDTH) + rec["product"]
            out.append(paint(head, "bold", color))
            out.append("")
            out.append(_labelled("You said", f'"{rec["said"]}"', width, color))
            out.append(_labelled("Brochure", _no_colons(rec["says"].strip()), width, color))
            continue
        note = body.startswith("(Note:") and body.endswith(")")
        if note:
            body = "Note: " + body[len("(Note:"):-1].strip()
        heading = not note and _is_heading(body)
        if heading:
            body = body[:-1]
        body = _no_colons(body)
        bullet = body.startswith("- ")
        pad = "  " + " " * lead
        wrapped = textwrap.fill(body, width, initial_indent=pad, subsequent_indent=pad + ("  " if bullet else ""))
        if color:
            wrapped = GLUED.sub(lambda m: paint(m.group(0), "cyan", True), wrapped)
            if heading:
                wrapped = paint(wrapped, "bold", True)
            if note:
                wrapped = paint(wrapped, "yellow", True)
        out.append(wrapped)
    return "\n".join(out).replace("\u00a0", " ")


STATUS_WORDS = {"answered": "answered", "abstained": "not in the data", "clarify": "needs a clearer question",
                "needs_info": "needs more information", "estimate": "estimate ready", "insufficient_data": "not enough past visits",
                "ok": "done", "invalid_arguments": "input rejected, asking again", "error": "failed"}


def format_tool_call(step: dict, color: bool = False) -> str:
    """One short line naming the tool the question was routed to, for example
         tool    product_info"""
    return paint(f"  tool    {step['name']}", "dim", color)


def format_footer(steps_line: str, usage: str | None, color: bool = False, seconds: float | None = None) -> str:
    parts = ["steps  " + steps_line] + ([f"{seconds:.1f} s"] if seconds is not None else []) + ([usage] if usage else [])
    return paint("  " + "   |   ".join(parts), "dim", color)


def rule(color: bool = False) -> str:
    return paint("-" * (line_width() + 2), "dim", color)


def prompt(color: bool = False) -> str:
    """The input prompt. The invisible-character markers keep line editing correct when the prompt is coloured."""
    return "\001\033[36m\002you>\001\033[0m\002 " if color else "you> "


class Spinner:
    """A one-line status while the assistant works, for example '/ Searching the brochures...'.
    It only animates in a real terminal. When output is piped it prints nothing, so logs stay clean."""
    FRAMES = "|/-\\"

    def __init__(self, stream=None, color: bool = False, interval: float = 0.1):
        self.stream = stream or sys.stdout
        self.enabled = self.stream.isatty()
        self.color, self.interval = color, interval
        self.text = "thinking"
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        """Print a line above the status line, so progress stays visible while the assistant keeps working."""
        with self._lock:
            self.stream.write(("\r\033[2K" if self.enabled else "") + text + "\n")
            self.stream.flush()

    def update(self, text: str) -> None:
        self.text = text

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            label = self.text[:1].upper() + self.text[1:]
            with self._lock:
                self.stream.write("\r\033[2K" + paint(f"{self.FRAMES[i % len(self.FRAMES)]} {label}...", "dim", self.color))
                self.stream.flush()
            i += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "Spinner":
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join()
            self.stream.write("\r\033[2K")  # clear the status line before the answer is printed
            self.stream.flush()
