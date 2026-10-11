"""Offline stand-in for a model API, for testing the harness only (never used for results).

It answers with schema-valid JSON whose values are placeholders (every scalar 1.0, uniform distributions), so a
mock answer can never be mistaken for physics. In tool conditions it first makes one python call. With
settings {"malformed_first": true} its first reply has no JSON, to exercise the parse-and-retry-once path.
"""
import json
from pathlib import Path

import yaml

from ai.harness.providers import Turn

ROOT = Path(__file__).resolve().parents[2]


def placeholder_observables():
    obs = yaml.safe_load(open(ROOT / "ai" / "observables.yaml"))
    bins = yaml.safe_load(open(ROOT / "config" / "bins.yaml"))["comparison"]
    out = {"method": "MOCK: placeholder values for harness tests", "data_sources": "none"}
    for o in obs["observables"]:
        if o["kind"] == "scalar":
            out[o["id"]] = 1.0
        elif o["kind"] == "fractions":
            out[o["id"]] = {c: 1.0 / len(o["categories"]) for c in o["categories"]}
        else:
            n = len(bins[o["grid"]]["edges"]) - 1
            out[o["id"]] = [1.0 / n] * n
    return out


def placeholder_events(n):
    ev = {"process": "compton", "e_in_MeV": 1.0, "e_out_MeV": 0.5, "cos_scatter": 0.0, "t_e_MeV": 0.5,
          "cos_electron": 0.5, "t_positron_MeV": None}
    return {"method": "MOCK: placeholder events for harness tests", "events": [dict(ev) for _ in range(n)]}


class MockConversation:
    def __init__(self, slot, system, user, tools):
        self.slot, self.tools, self.user = slot, tools, user
        self.settings = dict(slot.get("settings") or {})
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.calls = 0

    def step(self):
        self.calls += 1
        req = {"model": "mock", "messages": list(self.messages)}
        if self.tools and self.calls == 1:
            t = Turn("", [("mock_call_1", "python", {"code": "print(2 + 2)"})], {"input_tokens": 10, "output_tokens": 5},
                     "tool_use", req, {"mock": True}, "mock")
        elif self.settings.get("malformed_first") and not any(m.get("retry") for m in self.messages):
            t = Turn("I cannot format this.", [], {"input_tokens": 10, "output_tokens": 5}, "end", req,
                     {"mock": True}, "mock")
        else:
            body = placeholder_events(200) if "individual photon" in self.user else placeholder_observables()
            t = Turn("Mock answer.\n```json\n" + json.dumps(body) + "\n```", [],
                     {"input_tokens": 10, "output_tokens": 5}, "end", req, {"mock": True}, "mock")
        self.messages.append({"role": "assistant", "content": t.text, "tool_calls": t.tool_calls})
        return t

    def add_tool_results(self, results):
        for cid, _, out in results:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": out})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text, "retry": True})
