from app.core.background import strip_background


def test_who_someone_is_is_left_out_of_the_search():
    for line in ("The accused is from a Dalit family.", "The accused is a wealthy businessman.", "The accused is a poor daily-wage labourer.",
                 "The accused is a migrant worker from Bihar.", "The accused is 70 years old.", "He belongs to an upper-caste family."):
        clean, removed = strip_background(f"A man stole a bicycle. Which offence is this? {line}")
        assert clean == "A man stole a bicycle. Which offence is this?" and removed == [line]


def test_a_word_before_a_person_is_left_out():
    assert strip_background("A wealthy man hit a Dalit woman.")[0] == "A man hit a woman."
    assert strip_background("An upper-caste man slapped a tenant from Bihar.")[0] == "A man slapped a tenant."


def test_what_can_matter_in_law_stays():
    for text in ("The victim is 15 years old. He took her away.", "The accused is a public servant from Bihar. He took a bribe.",
                 "The accused is a poor clerk. He kept the money.", "He is poor and stole bread. What offence is this?"):
        assert strip_background(text) == (text, [])


def test_plain_text_is_returned_untouched():
    text = "A man  broke into a shed.\nWhat offence is this?"
    assert strip_background(text) == (text, [])


def test_nothing_is_removed_if_nothing_would_remain():
    assert strip_background("The accused is a wealthy businessman.") == ("The accused is a wealthy businessman.", [])
