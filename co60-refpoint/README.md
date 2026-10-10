# co60-refpoint

Photon-to-dose physics at the Co-60 reference point (central axis, 5 cm depth in water, 10 × 10 cm² field, SSD 100 cm): AI-generated answers compared with Monte Carlo at each step of the chain from incident photon to absorbed dose.

The specification is [STUDY_PLAN.md](STUDY_PLAN.md). Progress and open questions are in [STATUS.md](STATUS.md); design choices are in [DECISIONS.md](DECISIONS.md).

## Environment

Pinned versions are in `env/versions.lock.yaml`.

```
PREFIX=/opt/geant4 env/install_geant4.sh     # Geant4 11.4.3, multithreaded
PREFIX=/opt env/install_egsnrc.sh            # EGSnrc v2026 and the RZ user codes
python3 -m venv /opt/venv && /opt/venv/bin/pip install -r env/requirements.txt
source env/setup_env.sh
pytest
```

## Layout

| Path | Contents |
|---|---|
| `refs/refs.yaml` | every external number, with source and a `verified_by_humza` flag |
| `config/` | shared bin edges |
| `env/` | install scripts and version pins |
| `mc/geant4/`, `mc/egsnrc/` | study application and benchmark inputs |
| `mc/tools/` | run harnesses and manifest writer |
| `mc/runs/` | run manifests and outputs, never deleted |
| `ai/` | prompts, harness, raw responses |
| `analysis/` | analysis code, figures, tables |
| `tests/` | unit tests |
