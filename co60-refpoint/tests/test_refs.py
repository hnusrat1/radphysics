import yaml

from analysis.refs import REFS_PATH, load_refs

REQUIRED = {"key", "quantity", "value", "unit", "source", "entered_from", "verified_by_humza", "entered_on"}
ENTERED_FROM = {"plan", "source_code", "recall", None}


def test_keys_unique():
    with open(REFS_PATH) as f:
        keys = [e["key"] for e in yaml.safe_load(f)]
    assert len(keys) == len(set(keys))


def test_entries_complete():
    for key, entry in load_refs().items():
        missing = REQUIRED - entry.keys()
        assert not missing, f"{key} missing {missing}"
        assert entry["entered_from"] in ENTERED_FROM, key
        assert isinstance(entry["verified_by_humza"], bool), key
