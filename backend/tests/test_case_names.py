from types import SimpleNamespace

from app.explainability.case_names import check_case_names, party_words, sentences

J = lambda m, t: SimpleNamespace(marker=m, kind="judgment", title=t)  # noqa: E731
S = SimpleNamespace(marker=1, kind="statute", title="Punishment for robbery")
SOURCES = [S, J(2, "Ram Ratan v. State of Madhya Pradesh"), J(3, "Ganesan v. State Rep. By Station House Officer")]


def test_a_sentence_naming_another_judgment_is_flagged():
    answer = ("Robbery is punished under Section 392 [1]. In Ganesan v. State (SC, 2021), the court held that a "
              "deadly weapon must be used [2]. In Ram Ratan v. State of M.P. the court said so too [2].")
    flags = check_case_names(answer, SOURCES)
    assert len(flags) == 1 and flags[0].cited_marker == 2 and "Ganesan" in flags[0].named_case


def test_correct_and_unnamed_citations_pass():
    answer = "In Ganesan v. State, the court held this [3]. The court also noted the weapon was used [2]."
    assert check_case_names(answer, SOURCES) == []
    assert check_case_names("In Ganesan v. State [2].", [S, J(2, "Ram Ratan v. State")]) == []  # one judgment only


def test_helpers():
    assert party_words("Mohd. Rafiq v. State of Madhya Pradesh") == {"mohd", "rafiq"}
    assert sentences("In Mohd. Rafiq v. State of M.P. (SC, 2021), the court held X [3]. Next.") == [
        "In Mohd. Rafiq v. State of M.P. (SC, 2021), the court held X [3].", "Next."]
