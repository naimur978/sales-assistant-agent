"""Terminal formatting: wrapping, bullets, sources, notes, colours and the compact usage line. No network."""
import shutil

from src.display import format_footer, format_reply, format_tool_call, line_width, paint, use_color
from src.llm import Usage

LONG = "word " * 60


def narrow(monkeypatch, columns=70):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: shutil.os.terminal_size((columns, 24)))


def test_long_lines_are_wrapped_to_the_terminal_width(monkeypatch):
    narrow(monkeypatch)
    out = format_reply(LONG.strip())
    assert all(len(line) <= line_width() + 2 for line in out.split("\n")) and len(out.split("\n")) > 1


def test_bullets_get_a_hanging_indent(monkeypatch):
    narrow(monkeypatch)
    lines = format_reply("- " + LONG.strip()).split("\n")
    assert lines[0].startswith("  - ") and all(line.startswith("    ") for line in lines[1:])


def test_existing_indentation_and_blank_lines_are_kept(monkeypatch):
    narrow(monkeypatch)
    out = format_reply("Labour\n  labour: 6 h\n\nNext").split("\n")
    assert out[1].startswith("    labour") and out[2] == "" and out[3] == "  Next"


def test_the_partial_answer_note_is_marked_and_loses_its_brackets():
    out = format_reply("Answer.\n\n(Note: the brochures may say more than is shown here. Check the pages listed above before quoting this to the customer.)")
    assert "  Note - the brochures may say more than is shown here." in out and "(Note" not in out


def test_sources_are_dimmed_only_when_colour_is_on():
    text = "Fact. [page 2 of filter-optimizer.pdf]"
    assert "\033[" not in format_reply(text, color=False)
    assert "\033[36m[page 2 of filter-optimizer.pdf]\033[0m" in format_reply(text, color=True)


def test_colour_is_off_when_not_a_terminal_or_no_color_is_set(monkeypatch):
    class Pipe:
        def isatty(self):
            return False

    class Tty:
        def isatty(self):
            return True
    assert use_color(Pipe()) is False and use_color(Tty()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert use_color(Tty()) is False
    assert paint("x", "dim", False) == "x"


def test_tool_call_is_one_line_naming_the_tool():
    out = format_tool_call({"name": "product_info", "args": {"question": "Which Larox PF filter option helps?"}, "status": "answered"})
    assert out == "  tool    product_info"


def test_footer_shows_steps_and_usage_on_one_line():
    out = format_footer("product_info (answered)", "5 model calls, 7,977 tokens")
    assert "steps  product_info (answered)" in out and "5 model calls, 7,977 tokens" in out and "\n" not in out
    assert "5 model" not in format_footer("x", None)


def test_usage_line_has_no_cost_text_unless_prices_are_set(monkeypatch):
    from src import config
    u = Usage(calls=5, input_tokens=7319, output_tokens=658)
    assert u.short() == "5 model calls, 7,977 tokens"
    monkeypatch.setattr(config, "PRICE_PER_MILLION_TOKENS", {"input": 0.1, "output": 0.4})
    assert u.short().endswith("~$0.0010")


REC = ('Recommendation: Filter optimizer. You said "feed grade changes"; the brochure says: It keeps '
       "performance up. [page 2 of filter-optimizer.pdf]")


def test_a_recommendation_becomes_a_labelled_block():
    lines = format_reply(REC).split("\n")
    assert lines[0].startswith("  Recommendation  ") and lines[0].endswith("Filter optimizer")
    assert lines[1] == "" and lines[2].startswith('  You said        "feed grade changes"')
    assert lines[3].startswith("  Brochure        It keeps performance up.") and lines[3].endswith("[page 2 of filter-optimizer.pdf]")


def test_the_recommendation_line_is_bold_only_when_colour_is_on():
    assert format_reply(REC, color=True).startswith("\033[1m  Recommendation  Filter optimizer\033[0m")
    assert "\033[" not in format_reply(REC, color=False)


def test_a_recommendation_in_another_wording_is_left_as_it_is():
    out = format_reply("Recommendation: use the Filter optimizer because it helps.")
    assert out == "  Recommendation - use the Filter optimizer because it helps."


def test_section_titles_become_headings_without_the_colon():
    out = format_reply("Intro.\n\nOther relevant facts:\n- one\n- two")
    assert "  Other relevant facts" in out and "facts:" not in out
    assert "\033[1m  Other relevant facts\033[0m" in format_reply("Other relevant facts:\n- one", color=True)
    assert ":" not in format_reply("- a bullet that ends with a colon:")


def test_prompt_is_coloured_with_line_editing_markers_only_when_colour_is_on():
    from src.display import prompt
    assert prompt(False) == "you> "
    assert "\033[36m" in prompt(True) and "\001" in prompt(True) and prompt(True).endswith(" ")


def test_reworded_citations_are_shown_in_one_standard_wording():
    from src.display import normalise_citations
    assert normalise_citations("A [filter-optimizer, page 2]. B [larox-pf-ds, p.3]. C [larox-pf-ds, p 4].") == \
        "A [page 2 of filter-optimizer.pdf]. B [page 3 of larox-pf-ds.pdf]. C [page 4 of larox-pf-ds.pdf]."
    assert normalise_citations("Already [page 2 of x.pdf]") == "Already [page 2 of x.pdf]"
    assert "[page 2 of filter-optimizer.pdf]" in format_reply("Fact [filter-optimizer, page 2].")


def test_a_citation_is_never_split_across_two_lines(monkeypatch):
    narrow(monkeypatch, 60)
    out = format_reply("word " * 9 + "[page 2 of filter-optimizer.pdf] and more text after it", color=True)
    assert any("\033[36m[page 2 of filter-optimizer.pdf]\033[0m" in line for line in out.split("\n"))
    assert "\u00a0" not in out


# ---- spinner and timing
import io
import time

from src.display import Spinner


class FakeTty(io.StringIO):
    def isatty(self):
        return True


def test_spinner_prints_nothing_when_output_is_piped():
    out = io.StringIO()
    with Spinner(stream=out, interval=0.01) as sp:
        sp.update("searching")
        time.sleep(0.05)
    assert out.getvalue() == ""


def test_spinner_shows_the_current_step_and_clears_the_line_afterwards():
    out = FakeTty()
    with Spinner(stream=out, interval=0.01) as sp:
        time.sleep(0.04)
        sp.update("searching the brochures")
        time.sleep(0.06)
    shown = out.getvalue()
    assert "Thinking..." in shown and "Searching the brochures..." in shown
    assert shown.endswith("\r\033[2K")


def test_footer_shows_the_time_taken():
    assert "8.2 s" in format_footer("product_info (answered)", None, seconds=8.24)
    assert "|" not in format_footer("product_info (answered)", None)  # no time, no usage, so no separator


def test_no_colons_are_shown_in_a_terminal_answer():
    out = format_reply("Quick answer:\n- Fine.\n\nDetails:\n\nCaveat: one plant at 12:30.\n(Note: check the pages.)")
    assert ":" not in out.replace("12:30", "") and "Caveat - one plant" in out and "Note - check the pages." in out
