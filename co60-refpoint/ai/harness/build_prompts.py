#!/usr/bin/env python3
"""Build the versioned prompts and JSON schemas (STUDY_PLAN Section 5.3) from ai/observables.yaml,
ai/configs.yaml and config/bins.yaml.

    python -m ai.harness.build_prompts --version v1          # writes ai/prompts/v1/ and its MANIFEST.yaml
    python -m ai.harness.build_prompts --version v1 --check  # verifies the files match their MANIFEST hashes

Prompts contain the problem definition, the observables, units, bin edges and the output schema, and nothing
else: no Monte Carlo result and no reference value (STUDY_PLAN Section 10.2; tests/test_ai_prompts.py).
Once MANIFEST.yaml says frozen: true (set when M4 starts), the builder refuses to overwrite.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OBS = yaml.safe_load(open(ROOT / "ai" / "observables.yaml"))
CONFIGS = yaml.safe_load(open(ROOT / "ai" / "configs.yaml"))
BINS = yaml.safe_load(open(ROOT / "config" / "bins.yaml"))["comparison"]
CONDITIONS = ("A0", "A1", "A2")
A2_FILES = {
    "xcom_water.txt": "photon interaction coefficients of liquid water by process (coherent, incoherent, "
                      "photoelectric, pair production in the nuclear and electron fields), and the total, "
                      "in cm^2/g, versus photon energy in MeV",
    "estar_water.txt": "electron collision, radiative and total stopping powers of liquid water in MeV cm^2/g, "
                       "CSDA range in g/cm^2, radiation yield and density-effect parameter, versus electron "
                       "kinetic energy in MeV",
}


def grid_edges(name):
    return BINS[name]["edges"]


def fmt_edges(edges):
    return ", ".join(f"{e:g}" for e in edges)


def schema_observables():
    props = {"method": {"type": "string"}, "data_sources": {"type": "string"}}
    for o in OBS["observables"]:
        if o["kind"] == "scalar":
            props[o["id"]] = {"type": "number"}
        elif o["kind"] in ("spectrum", "distribution"):
            n = len(grid_edges(o["grid"])) - 1
            props[o["id"]] = {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}
        elif o["kind"] == "fractions":
            props[o["id"]] = {"type": "object", "additionalProperties": False,
                              "required": o["categories"],
                              "properties": {c: {"type": "number"} for c in o["categories"]}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "required": list(props), "additionalProperties": False, "properties": props}


def schema_events():
    item = {"type": "object", "additionalProperties": False, "required": [], "properties": {}}
    for f in OBS["events"]["fields"]:
        item["required"].append(f["id"])
        if f["type"] == "string":
            item["properties"][f["id"]] = {"type": "string", "enum": f["enum"]}
        elif f["type"] == "number":
            item["properties"][f["id"]] = {"type": "number"}
        else:
            item["properties"][f["id"]] = {"type": ["number", "null"]}
    n = OBS["events"]["n_per_call"]
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "required": ["method", "events"], "additionalProperties": False,
            "properties": {"method": {"type": "string"},
                           "events": {"type": "array", "items": item, "minItems": n, "maxItems": n}}}


def config_block(cfg):
    c = CONFIGS[cfg]
    return "\n".join([
        f"- Phantom: {c['phantom']}.",
        f"- Outside the phantom: {c['outside']}.",
        f"- Source: {c['source']}.",
        f"- Photon energies: {c['energies']}.",
        f"- Point of interest: {c['point']}.",
        f"- Scoring volume: {c['scoring_volume']}.",
        f"- Normalization: {c['phi0']}.",
        f"- Depth dose: {c['pdd']}.",
    ])


DEFINITIONS = """Definitions:
- Primary photon: a photon that has not interacted since leaving the source. Scattered photon: every other
  photon (scattered source photons and secondary photons such as bremsstrahlung, fluorescence and annihilation
  photons).
- theta: the angle between a particle's direction and the beam axis (the direction from the source towards the
  phantom), so cos(theta) = 1 is straight down the beam axis.
- Fluence: track length per unit volume in the scoring volume. Spectra are fluence per energy bin (not per MeV),
  divided by Phi0.
- Delta rays: electrons set in motion by electrons. They are part of the electron fluence (Layer 4) and not of
  the electrons set in motion by photon interactions (Layer 3).
- Quantities "at the point" refer to the scoring volume."""


def grids_block():
    return "\n".join([
        "Bin edges (MeV for energies; each list gives the edges, so a list of n+1 numbers defines n bins):",
        f"- photon_energy ({len(grid_edges('photon_energy')) - 1} bins): {fmt_edges(grid_edges('photon_energy'))}",
        f"- electron_energy ({len(grid_edges('electron_energy')) - 1} bins; the last bin is wider): "
        f"{fmt_edges(grid_edges('electron_energy'))}",
        f"- cos_theta ({len(grid_edges('cos_theta')) - 1} bins): {fmt_edges(grid_edges('cos_theta'))}",
    ])


def observables_block():
    out = []
    layers = {1: "Layer 1. Photons arriving at the point", 2: "Layer 2. Photon interactions at the point",
              3: "Layer 3. Electrons set in motion", 4: "Layer 4. Electrons slowing down",
              5: "Layer 5. Energy deposited"}
    for L in range(1, 6):
        out.append(layers[L])
        for o in [o for o in OBS["observables"] if o["layer"] == L]:
            shape = {"scalar": "number", "fractions": "object with keys " + ", ".join(o.get("categories", [])),
                     "spectrum": f"list of {len(grid_edges(o.get('grid', 'cos_theta'))) - 1} numbers on the "
                                 f"{o.get('grid')} grid",
                     "distribution": f"list of {len(grid_edges(o.get('grid', 'cos_theta'))) - 1} numbers on the "
                                     f"{o.get('grid')} grid, summing to 1"}[o["kind"]]
            out.append(f"- {o['id']} [{o['unit']}; {shape}]: {' '.join(o['text'].split())}")
        out.append("")
    return "\n".join(out).rstrip()


def task_observables(cfg):
    return f"""We are studying how photons and electrons deposit energy at one point in a water phantom. Report the
quantities below for this configuration.

Configuration:
{config_block(cfg)}

{DEFINITIONS}

{grids_block()}

Quantities to report (JSON key, unit and shape in brackets):
{observables_block()}

Give a value for every quantity. If you are unsure, give your best estimate rather than leaving it out.
Also give "method": a short description of how you obtained the values (at most 150 words), and
"data_sources": the cross-section and stopping-power data you relied on.

Answer format: end your reply with one JSON object in a ```json fenced code block, with exactly the keys above
plus "method" and "data_sources", using the units stated. The JSON schema is:
```json
{json.dumps(schema_observables(), indent=1)}
```"""


def task_events(cfg):
    n = OBS["events"]["n_per_call"]
    fields = "\n".join(f"- {f['id']}: {f['text']}" + (f" (one of: {', '.join(f['enum'])})" if "enum" in f else "")
                       for f in OBS["events"]["fields"])
    return f"""We are studying how photons interact at one point in a water phantom. Return {n} individual photon
interactions that occur in the scoring volume of this configuration, as an unbiased random sample of all photon
interactions there (primary and scattered photons, every process, in the proportions in which they occur).

Configuration:
{config_block(cfg)}

{DEFINITIONS}

Fields of each event (energies in MeV):
{fields}

Also give "method": a short description of how you generated the events (at most 100 words).

Answer format: end your reply with one JSON object in a ```json fenced code block with the keys "method" and
"events" (a list of exactly {n} events). The JSON schema is:
```json
{json.dumps(schema_events(), indent=1)}
```"""


def system_prompt(condition):
    base = "You are an expert in radiation physics and radiation dosimetry. Answer the question you are given."
    if condition == "A0":
        return base + " Work from your own knowledge; you have no tools."
    tool = (" You have a tool, python, that runs Python 3 code and returns its standard output and standard error."
            " numpy and scipy are installed; there is no internet access and no radiation transport code is"
            " installed. Each call may run for at most 10 minutes on 4 CPU cores. Files you write in the working"
            " directory persist between calls.")
    if condition == "A1":
        return base + tool
    files = "; ".join(f"/tmp/data/{k}: {v}" for k, v in A2_FILES.items())
    return base + tool + f" These data files are available (read-only; each has a header describing its columns): {files}."


def build(version, check=False):
    out = ROOT / "ai" / "prompts" / version
    manifest_path = out / "MANIFEST.yaml"
    files = {f"system_{c}.txt": system_prompt(c) for c in CONDITIONS}
    for cfg in CONFIGS:
        files[f"task_observables_{cfg}.txt"] = task_observables(cfg)
        files[f"task_events_{cfg}.txt"] = task_events(cfg)
    files["schema_observables.json"] = json.dumps(schema_observables(), indent=1) + "\n"
    files["schema_events.json"] = json.dumps(schema_events(), indent=1) + "\n"
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(files.items())}
    if check:
        m = yaml.safe_load(open(manifest_path))
        bad = [k for k, h in hashes.items() if m["sha256"].get(k) != h
               or hashlib.sha256((out / k).read_bytes()).hexdigest() != h]
        return bad
    if manifest_path.exists() and yaml.safe_load(open(manifest_path)).get("frozen"):
        sys.exit(f"{out} is frozen (M4 started); a change needs a new version and a DECISIONS entry.")
    out.mkdir(parents=True, exist_ok=True)
    for k, v in files.items():
        (out / k).write_text(v)
    manifest = {"version": version, "frozen": False,
                "built_from": ["ai/observables.yaml", "ai/configs.yaml", "config/bins.yaml",
                               "ai/harness/build_prompts.py"],
                "sha256": hashes}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return list(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    r = build(a.version, a.check)
    if a.check:
        print("prompts match MANIFEST" if not r else f"MISMATCH: {r}")
        sys.exit(1 if r else 0)
    print(f"wrote {len(r)} files to ai/prompts/{a.version}/")


if __name__ == "__main__":
    main()
