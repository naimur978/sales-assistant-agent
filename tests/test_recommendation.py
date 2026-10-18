"""The recommendation step: product comes from the cited file, the customer phrase must be the user's own words,
and the claim must have survived verification. Fake search and fake model output; no network."""
import pytest

from src import grounded
from src.grounded import ExtractedClaim, Extraction, Recommendation, answer

OPT_PAGE = ("Optimized filter performance even under changing process conditions. "
            "Each cake is measured after discharge and new setpoints calculated.")
DS_PAGE = "The Larox PF-DS is a fully automatic tower press filter with consistent results under varying process conditions."
USER = "We have Larox pressure filters but performance drops whenever our feed grade changes"


class FakeSearch:
    pages = {"filter-optimizer:2": OPT_PAGE, "larox-pf-ds:2": DS_PAGE}

    def search(self, query, k):
        hits = [{"id": i, "file": i.split(":")[0], "page": int(i.split(":")[1]), "text": t} for i, t in self.pages.items()]
        return {"hits": hits, "gate_score": 0.9}


OPT = ExtractedClaim(claim="The Filter optimizer keeps filter performance up under changing process conditions.",
                     quote="Optimized filter performance even under changing process conditions", file="filter-optimizer", page=2)
DS = ExtractedClaim(claim="The Larox PF-DS gives consistent results under varying process conditions.",
                    quote="consistent results under varying process conditions", file="larox-pf-ds", page=2)
BAD = ExtractedClaim(claim="The optimizer reduces energy by 40%.", quote="reduces energy consumption by 40 percent",
                     file="filter-optimizer", page=2)  # quote is not on the page


@pytest.fixture
def run(monkeypatch):
    def _run(extraction, entail_all=True):
        monkeypatch.setattr(grounded, "_generate", lambda *a, **k: extraction)
        monkeypatch.setattr(grounded, "entailed", lambda cl: set(range(len(cl))) if entail_all else set())
        return answer("Which product helps?", FakeSearch(), also=USER)
    return _run


def test_valid_recommendation_leads_the_answer_with_product_from_file_name(run):
    a = run(Extraction(claims=[DS, OPT], recommendation=Recommendation(
        claim_index=1, customer_phrase="performance drops whenever our feed grade changes")))
    assert a.status == "answered" and a.recommended == "Metso Filter optimizer"
    first = a.text.splitlines()[0]
    assert first.startswith("Recommendation: Metso Filter optimizer.") and "[filter-optimizer, p.2]" in first
    assert "Other relevant facts:" in a.text and "[larox-pf-ds, p.2]" in a.text


def test_phrase_not_in_users_words_is_rejected(run):
    a = run(Extraction(claims=[OPT], recommendation=Recommendation(claim_index=0, customer_phrase="costs are too high")))
    assert a.recommended is None and a.text.startswith("- ")  # facts still given, no recommendation


def test_recommendation_must_point_at_a_verified_claim(run):
    a = run(Extraction(claims=[BAD, OPT], recommendation=Recommendation(
        claim_index=0, customer_phrase="performance drops whenever our feed grade changes")))
    assert a.recommended is None and any("not found verbatim" in why for _, why in a.dropped)


def test_index_follows_the_claim_even_after_an_earlier_claim_is_dropped(run):
    a = run(Extraction(claims=[BAD, OPT], recommendation=Recommendation(
        claim_index=1, customer_phrase="performance drops whenever our feed grade changes")))
    assert a.recommended == "Metso Filter optimizer"


def test_out_of_range_index_is_ignored(run):
    a = run(Extraction(claims=[OPT], recommendation=Recommendation(claim_index=7, customer_phrase="feed grade changes")))
    assert a.recommended is None


def test_one_word_phrase_is_not_enough(run):
    a = run(Extraction(claims=[OPT], recommendation=Recommendation(claim_index=0, customer_phrase="performance")))
    assert a.recommended is None


def test_no_recommendation_when_model_gives_none(run):
    a = run(Extraction(claims=[OPT], recommendation=None))
    assert a.status == "answered" and a.recommended is None


def test_nothing_survives_means_abstain_not_recommend(run):
    a = run(Extraction(claims=[OPT], recommendation=Recommendation(
        claim_index=0, customer_phrase="feed grade changes")), entail_all=False)
    assert a.status == "abstained" and a.recommended is None
