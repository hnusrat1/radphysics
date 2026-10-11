# AI harness (M3) and how M4 is launched

| Path | What it is |
|---|---|
| `observables.yaml` | The quantities an answer reports, with units, grids, the Monte Carlo key each is scored against, the tolerance class and the hypothesis tags. It is the single source for the prompts, the schema and the scorer. |
| `configs.yaml` | R0 and the pilot P0, as inputs only |
| `prompts/v1/` | Frozen prompt files and JSON schemas, built by `harness/build_prompts.py`; `MANIFEST.yaml` holds their SHA-256 |
| `harness/sandbox.py` | Isolated Python for A1/A2: no network, the repository, transport codes and transcripts hidden, own PID space, 4 cores, 10 min per call |
| `harness/providers.py` | Anthropic, OpenAI, Gemini and OpenAI-compatible (open-weight hosts) clients over HTTPS, with one tool loop |
| `harness/run.py` | The run matrix. Every conversation is written once, read-only, to `raw/<tag>/<slot>/` |
| `harness/mock.py` | Offline placeholder model for tests; never used for results |
| `data/` | The A2 tables (see `data/README.md`) |
| `models.yaml`, `prices.yaml` | Model slots and the cost model. Model IDs are filled in and `frozen: true` is set at launch |

Rules enforced in code:
- No reference value in a prompt (`tests/test_ai_prompts.py`).
- Prompts hash-checked before every run.
- R0 runs refused until `models.yaml` and the prompt manifest are frozen.
- One logged retry on malformed JSON.
- Refusals, parse failures, API errors and tool calls are counted per conversation.

## Launch checklist (Humza)

1. **Keys.** Add the API keys as environment secrets (cloud environment → Edit → Network secrets): `ANTHROPIC_API_KEY`,
   `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `OPENROUTER_API_KEY` for the open-weight host. Start a new session so they are
   picked up.
2. **Network.** Allow `api.anthropic.com`, `api.openai.com`, `generativelanguage.googleapis.com` and `openrouter.ai`
   (Network access → Allowed domains).
3. **Models.** Fill `model_id` for each slot in `models.yaml`: the most capable generally available model per frontier
   provider, and two pinned open-weight releases from different labs (D028 item 5). Add each provider's highest
   reasoning setting under `settings`.
4. **Pilot.** Run P0 (excluded from analysis):
   `python -m ai.harness.run --tag P0-pilot --config P0 --k 1`.
   Fix formatting problems only. Then replace `tokens_per_conversation` in `prices.yaml` with the measured usage, enter
   official prices, and rerun `--dry-run` for the M4 estimate (current estimate: about $130, high case about $400).
5. **Approve.** Approve the budget (STUDY_PLAN 10.5). Set `frozen: true` in `models.yaml` and in `prompts/v1/MANIFEST.yaml`,
   and record both in DECISIONS.
6. **Run M4.**
   `python -m ai.harness.run --tag M4-R0 --config R0 --k 5`.
   Then score with `python analysis/reference.py && python analysis/m5.py --tag M4-R0`.
