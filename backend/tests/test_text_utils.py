from app.core.text_utils import split_oversized, split_sentences


def test_does_not_split_on_legal_abbreviations() -> None:
    text = "In State v. Kumar the court applied S. 304A of the Code. The appeal failed."
    assert split_sentences(text) == [
        "In State v. Kumar the court applied S. 304A of the Code.",
        "The appeal failed.",
    ]


def test_does_not_split_on_initials_or_amendment_notes() -> None:
    text = "A. strikes Z. with a stick. Subs. by Act 26 of 1955, w.e.f. 1-1-1956. Next sentence."
    assert split_sentences(text) == [
        "A. strikes Z. with a stick.",
        "Subs. by Act 26 of 1955, w.e.f. 1-1-1956.",
        "Next sentence.",
    ]


def test_split_oversized_prefers_clause_punctuation() -> None:
    sentence = "; ".join(["clause number %d with some words" % i for i in range(10)]) + "."
    pieces = split_oversized(sentence, 80)
    assert all(len(p) <= 80 for p in pieces)
    assert " ".join(pieces).replace("; ", ";").replace(" ", "") == sentence.replace("; ", ";").replace(" ", "")
