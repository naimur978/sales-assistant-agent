

def test_a_customer_limit_must_be_something_the_user_said():
    from src.slots import ungrounded
    said = "Our target is under 2.5% silica and over 65% iron"
    assert ungrounded({"silica_max": 2.5, "iron_min": 65.0}, said) == []
    assert ungrounded({"silica_max": 1.8}, said) == ["silica_max"]
