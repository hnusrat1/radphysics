"""Read external values from refs/refs.yaml by key."""

from pathlib import Path

import yaml

REFS_PATH = Path(__file__).resolve().parent.parent / "refs" / "refs.yaml"


def load_refs(path=REFS_PATH):
    with open(path) as f:
        entries = yaml.safe_load(f)
    return {e["key"]: e for e in entries}


def ref_value(key, path=REFS_PATH):
    """Return the value for key. Raises if the entry has no value yet."""
    entry = load_refs(path)[key]
    if entry["value"] is None:
        raise ValueError(f"refs.yaml entry {key!r} has no value yet")
    return entry["value"]
