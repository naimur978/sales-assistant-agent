from src.verifier import Claim, verify_claim

PAGES = {
    "larox-pf-ds:2": (
        "The Larox PF-DS tower press filter plate pack is composed of polypropylene\n"
        "membrane filter plates. Depending on the number of chambers, the total\n"
        "filtration area is 38–94 m² per unit, with an operating pressure up to 16 bar."
    ),
    "larox-pf-brochure-a:2": "With over 1,300 installations around the world. Filter areas of 1.6, 2.5 and 6 m2.",
}


def c(claim, quote, file="larox-pf-ds", page=2):
    return Claim(claim=claim, quote=quote, file=file, page=page)


def ok(claim):
    return verify_claim(claim, PAGES)[0]


def test_valid_claim_passes():
    assert ok(c("The PF-DS operates at up to 16 bar.", "with an operating pressure up to 16 bar"))


def test_quote_survives_line_breaks_and_case():
    assert ok(c("Plates are polypropylene.", "Polypropylene MEMBRANE filter\nplates"))


def test_quote_not_on_page_rejected():
    assert not ok(c("It handles 25 bar.", "with an operating pressure up to 25 bar"))


def test_made_up_page_rejected():
    assert not ok(c("Up to 16 bar.", "with an operating pressure up to 16 bar", page=9))


def test_made_up_file_rejected():
    assert not ok(c("Up to 16 bar.", "with an operating pressure up to 16 bar", file="larox-pf-xx"))


def test_quote_from_wrong_page_rejected():
    assert not ok(c("1,300 installations.", "With over 1,300 installations around the world", page=2))


def test_number_not_in_quote_rejected():
    assert not ok(c("Operating pressure is up to 20 bar.", "with an operating pressure up to 16 bar"))


def test_number_in_claim_must_come_from_quote_not_page():
    # 94 is on the page but not in the quote
    assert not ok(c("Filtration area is up to 94 m².", "Depending on the number of chambers"))


def test_decimal_16_is_not_1_6():
    assert not ok(c("Plate area is 16 m2.", "Filter areas of 1.6, 2.5 and 6 m2", "larox-pf-brochure-a", 2))
    assert ok(c("Plate area is 1.6 m2.", "Filter areas of 1.6, 2.5 and 6 m2", "larox-pf-brochure-a", 2))


def test_thousands_separator_normalised():
    assert ok(c("Over 1300 installations.", "With over 1,300 installations around the world", "larox-pf-brochure-a", 2))


def test_too_short_quote_rejected():
    assert not ok(c("It has a bar.", "16 bar"))


def test_partial_word_not_matched():
    assert not ok(c("Plates are poly.", "composed of poly membrane"))


def test_stitched_quote_rejected():
    # two real fragments joined into a sentence the brochure never says
    assert not ok(c("Polypropylene plates reach 16 bar.", "polypropylene membrane filter plates up to 16 bar"))


def test_model_number_in_claim_is_not_a_fact():
    assert ok(c("The PF-DS 16 operating pressure is up to 16 bar.", "with an operating pressure up to 16 bar"))
    assert ok(c("Larox PF 60 Model 2025 works at up to 16 bar.", "with an operating pressure up to 16 bar"))


def test_model_number_exemption_does_not_hide_a_wrong_value():
    assert not ok(c("The PF-DS 150 bar limit applies.", "with an operating pressure up to 16 bar"))
    assert not ok(c("Larox PF 60 works at up to 20 bar.", "with an operating pressure up to 16 bar"))
    assert not ok(c("Larox PF-DS 25 bar.", "with an operating pressure up to 16 bar"))
