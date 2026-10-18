from types import SimpleNamespace

from app.core.bns import annotate, bns_for, source


def test_known_ipc_to_bns_mappings_from_the_official_table():
    assert bns_for("302") == ["103"]  # murder
    assert bns_for("379") == ["303"]  # theft
    assert bns_for("420") == ["318"]  # cheating
    assert bns_for("304b") == ["80"]  # dowry death (case-insensitive)
    assert bns_for("34") == ["3(5)"]  # common intention
    assert bns_for("124A") == []  # sedition: no direct equivalent
    assert "Bureau of Police Research and Development" in source()


def test_annotate_marks_statute_sources_only():
    statute = SimpleNamespace(kind="statute", section="302", bns=[])
    repealed = SimpleNamespace(kind="statute", section="497", bns=[])
    judgment = SimpleNamespace(kind="judgment", section="", bns=[])
    annotate([statute, repealed, judgment])
    assert statute.bns == ["103"] and repealed.bns == ["none"] and judgment.bns == []
