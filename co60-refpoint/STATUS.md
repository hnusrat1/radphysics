# Status

Humza delegated every open choice on 2026-10-11: "pick your best recommended choices, continue and dont stop until you have the full study ready for me to review" (DECISIONS D028). Under that delegation M2 and M3 run back to back, without a gate stop between them. Two things stay with Humza under Section 10: verifying `refs.yaml`, and approving paid API spend. M4 ran on 2026-10-12 through one OpenRouter key (D029, D030); everything since has cost nothing.

## M4. AI runs (2026-10-12). R0 complete and scored.

- **Results.** `analysis/results/M5_M4-R0.json` (scorer `analysis/m5.py`) and the page
  `analysis/results/co60_m4_results.html` (`analysis/m4_page.py`). All 125 R0 conversations have an outcome; no
  API error remains.
  - D/Φ₀ within 2% with Python (A1 and A2): Claude Haiku 5.5 (−0.3%, −0.6%) and DeepSeek V4 Flash (−0.3%, +1.0%).
    Both pass every Layer 5 dose quantity in both tool conditions.
  - Without tools (A0), no model passes D/Φ₀. Haiku −2.6%, Qwen +5%, GPT-6 Luna −9%, DeepSeek −18%, and Gemini
    about ×3,000 (wrong unit).
  - GPT-6 Luna stays 7–11% low with tools; Qwen moves further off (+17%, +30%). Gemini made no Python call on the
    observables task, answered with zeros twice in A1, and was off by a unit factor in A2.
  - Every model and condition breaks at Layer 1: spectral and angular shapes fail the 3 × half-sample criterion,
    even where the scalars pass.
  - Events, bound-electron rule: Haiku A0 0.2% and A1 0% invalid, DeepSeek A0 0% and A1 13%, Qwen A1 51%, Gemini
    28–66%. Haiku A1 and GPT-6 Luna's single answered A1 call match the Monte Carlo distribution. GPT-6 Luna
    refused 9 of 10 event calls.
- **Cost (measured OpenRouter charge).** R0 $7.78. P0 pilot $4.19, plus $0.38 for the cached-Haiku test and $0.13
  for a cut DeepSeek conversation. Void attempts $0.005. Under Humza's $10 limit for R0 (D032).
- **Models (D030).** The fast, low-cost tier of each provider. Haiku runs with prompt caching (D032).
- **Access.** The environment's network secret for `openrouter.ai` first sent the key under a header named
  `OPENROUTER_API_KEY`, which OpenRouter ignores (HTTP 401). With `Authorization: Bearer <key>` it worked.
- **Pilot fixes (D031) and harness issues found during R0 (D033 to D035):**
  - in-band OpenRouter errors are now retried, and Gemini's four API-error conversations were rerun;
  - R0 ran as parallel processes per block;
  - the parser fallback on malformed fenced JSON was fixed after R0. Re-scoring with the fix gives identical
    results.
- **Environment.** `/opt/venv` is rebuilt from `env/requirements.txt`. `test_a2_sandbox_sees_data_and_not_the_repository`
  failed in the M4 container only because `/opt/geant4` did not exist there; it no longer assumes the path exists.

- **Code audit (D036, exploratory).** All 50 A1/A2 observables conversations were traced to their code.
  - The photon field decides the answer. The 17 answers built on a simulation the model wrote have median errors of
    0.8% in the scattered fluence and 0.5% in D/Φ₀. The 20 with an assumed scatter field are off by 57% and 16%, and
    sit on the line expected when only the scatter is wrong (scattered photons give 18.8% of the dose).
  - Simulated runs per model: Haiku 10 of 10, DeepSeek 8 (2 with major bugs), GPT-6 Luna 2 (a C++ run as accurate
    as Haiku, and one with a bug), Qwen 0. Gemini wrote no code.
  - 19 of 20 simulations do not transport electrons, none models electron binding, and their photoelectric share
    is about a third of the Monte Carlo's.
  - 11 of 40 answers come entirely from the code's output, and 11 stated methods describe calculations that are
    not in the code.
  - The sandbox exposed `g++` and `clang`; GPT-6 Luna compiled C++ in three R0 conversations.

### After the runs (2026-10-29)

Done without any API call; nothing was spent.

- **Merge.** The M4 branch is merged into the M2-final branch.
  - Re-scoring M4-R0 against the final reference gives identical verdicts.
  - The two floors added after the M4 branch forked (β and PDD(5)) change no verdict.
- **Compilers removed from the sandbox (D037).**
  - Hidden: the GCC and LLVM back ends, the assembler and linkers, Go, Java, Node and the browsers. A test checks
    that every compiler fails and that scipy runs.
  - The three compiled conversations are found from the code the model ran. Without them, no breakpoint changes.
  - GPT-6 Luna A2's median dose error goes from −10.6% to −13.2% without them. Its best dose answer (A2 r2, +0.1%)
    and its only answered events call are both compiled ones.
- **Revised shape criterion (D038, exploratory for M4-R0).** A distribution passes when W1 is within 5% of the
  Monte Carlo mean energy, or within 0.1 in cos θ. JSD is reported only.
  - The controls behave as under the pre-set criterion:
    - Livermore and Penelope show no breakpoint (Penelope's one unresolved shape now passes);
    - the +5% dose control breaks at Layer 5;
    - the uniform answer breaks at Layer 1.
  - Haiku A1 and A2 break at Layer 2.
    - Layer 2 fails on the photoelectric share of interactions, 0.7% against 2.4%.
    - Layer 4 fails on the electron angular distribution.
    - Every Layer 1 and Layer 3 quantity and D/Φ₀ pass.
  - Every other model and condition stays at Layer 1, mostly on the scattered-photon spectrum.
  - By photon field (D036), the median W1 of the scattered spectrum is 5 keV for simulated answers and 218 keV for
    assumed ones. The tolerance is 24 keV.
- **H3 caveat (D040).** The PDD(5) reference is for model A, which has no head scatter. A measured table can differ
  from it by an amount not known until a published PDD(5) is in refs.yaml, so H3 is exploratory until then.
  - A0 answers against the reference: GPT-6 Luna −1.6%, Haiku −2.8%, DeepSeek +5.2%, Qwen −17%, Gemini −33%.
- **What failed or was anomalous.**
  - **The pre-set shape criterion cannot be met by an answer that is not itself a Monte Carlo** of the reference's
    precision (D028 item 4).
    - This is a design error, found only after the data.
    - D038 is the fix for later runs. M4-R0 keeps the pre-set criterion as its primary analysis.
  - **The first compiler regex missed `'g++'` inside a Python list,** because `\b` finds no word boundary after `+`.
    - It was fixed before any result was written.
    - Detection now matches D036's audit exactly: three R0 conversations and one pilot conversation.

### Next
- Humza's review of the results page and D030 to D040.
- Paid options, each only with Humza's approval (Section 10.5):
  - rerun the three compiled conversations without compilers, about $0.10;
  - a second configuration with no tabulated answer to recall (R1), about $8 for the AI side; its Monte Carlo is
    free;
  - a higher tier of the same providers, about $65; the top tier about $240 for the full matrix, or about $160 for
    A0 and A1 only.

## M2. Monte Carlo validation (2026-10-11 to 12). Complete.

### Summary

The Monte Carlo reference for the AI comparison is final. It is option4 at msc range factor 0.04, with the dose scored by position away from any boundary. Two other Geant4 lists and EGSnrc agree with it well inside the AI tolerances.

- **R0 reference:**
  - D/Φ₀ = 5.3672 × 10⁻¹² Gy cm² ± 0.090% (boundary-free, 45 batches).
  - β = 1.0053 ± 0.0014.
  - s_w,air = 1.12678 (ICRU 90 basis); 1.13311 on ICRU 37.
  - PDD(5) = 81.79 ± 0.22%.
  - All layers come from `M2-R0-option4`.
- **Step size.** Range factor 0.04 against 0.08:
  - The dose changes by −0.08% ± 0.13%, and no other physical quantity moves beyond 1.6σ.
  - Both step-size Fano diagnostics pass.
  - The scoring-cylinder boundary artifact is −0.14% ± 0.13% for option4 and +0.02% ± 0.20% for both other lists. At range factor 0.08 it was −0.24% ± 0.11%, so the same step limit caused both the boundary artifact and the Fano deviation.
- **Physics lists.**
  - Boundary-free dose: Livermore +0.09% ± 0.18%, Penelope −0.01% ± 0.17%.
  - β: 1.0053 (option4), 1.0054 (Livermore), 1.0032 (Penelope).
  - Every scalar observable agrees within 0.8%. The resolved differences come from Penelope's stopping power (electron fluence −0.8%) and Livermore's Compton model (interaction shares +0.5%).
- **EGSnrc in B0.** Dose −0.20% ± 0.17%, s_w,air +0.008%, photon fluence +0.14%, electron fluence +0.46%. All are inside the 0.5% investigation threshold (Section 4.2).
- **B0 spectrum shapes** (added 2026-10-29; 10 keV bins above 10 keV, EGSnrc against Geant4; `b0.spectra_metrics` in `M2_results.json`):
  - **Photon fluence.** Mean energy −0.04% ± 0.03%. W1 is 0.36 keV, against 0.24 to 0.25 keV between half-samples of either code.
  - **Photon bins.** 1 of 116 bins lies beyond 3σ, about what chance gives.
  - **Electron fluence: a resolved shape difference at low energy.**
    - The mean energy is −0.41% ± 0.08% (EGSnrc lower, 5σ), inside the 0.5% threshold. W1 is 1.8 keV, against 1.1 keV between EGSnrc's half-samples.
    - 29 of 111 bins lie beyond 3σ. The difference is largest in the lowest bin: EGSnrc has 6.0% ± 0.2% more fluence at 10–20 keV than Geant4's CSDA-weighted estimator (D020), and about 1.7% more at 20–50 keV.
    - Geant4's linear estimator is further off: 17% at 10–20 keV. The CSDA-weighted estimator is the closer of the two, which supports D020 but does not close the gap.
    - Likely cause: FLURZnrc spreads each step's path length evenly in energy. That gives the low-energy end of each step more path length than Geant4's 1/S(E) weighting does, the direction seen, and it inflates the bin just above EGSnrc's 10 keV cut-off by 16.5% (D039).
  - **Effect on the reference.** Dose and s_w,air in B0 agree regardless (−0.20% and +0.008%).
    - This B0 W1 of 1.8 keV is larger than the resolution floor for the R0 electron spectrum (0.7 keV, from the Geant4 lists).
    - Raising that floor to 1.8 keV would change none of the 62 scored AI answers, so the floor stays as defined.
    - Under D038 the tolerance for that spectrum is 22 keV.
- **s_w,air calculation against SPRRZnrc (required test, D039; added 2026-10-29).**
  - With EGSnrc's own fluence (FLURZnrc, 2 keV bins), the same PEGS4 stopping powers, and the track-end fluence extrapolated from 12–30 keV, our calculation reproduces SPRRZnrc to +0.0022% ± 0.0018% (1.2σ). The plan's criterion is met at k = 2.
  - Reading the track-end fluence from the first bin above the cut-off instead gives −0.0056% (3.0σ); FLURZnrc inflates that bin. The extrapolation was chosen after both were seen, so the 0.008% between them counts as the check's systematic uncertainty.
  - The reference value, which uses Geant4's fluence, is 0.0079% ± 0.0016% below SPRRZnrc. Every variant agrees within 0.01%, two hundred times below the 2% tolerance on s_w,air.
- **Published values.**
  - s_w,air on the ICRU 37 basis is 1.1331, against TRS-398's 1.133.
  - Geant4's tracking μ agrees with the XCOM data within 0.03%.
  - Both external values are unverified (Section 10.1).
  - Still missing from refs.yaml: NIST μ/ρ and μ_en/ρ tables, a published PDD(5), and a numeric β.
- **Resolution floors** for the AI scoring now exist for every scalar: the largest list or EGSnrc difference plus 2σ. Examples: D/Φ₀ 0.54%, β 0.71%, electron fluence 1.1%, s_w,air 0.011%, PDD(5) 0.54% (its own 2σ, one scan only). All are below the 2% and 5% tolerances.

### Gate (M2, STUDY_PLAN Section 9)

The plan's M2 gate is: Humza verifies every reference value in `refs.yaml` and accepts the Monte Carlo as the reference. The verification is his (Section 10.1), so it stays open. Under D028 I treated the Monte Carlo as accepted for building and testing M3/M5, and M4-R0 is now scored against it.

### Progress
- **Fano step-size diagnostics: done.** Range factor 0.04 gives 1.00001 ± 0.00088 and step function 0.05 / 1 µm gives 0.99921 ± 0.00073, against 1.00157 ± 0.00058 at production stepping. M2 runs at range factor 0.04 (D026, D028). The batch-0 physics printout of every M2 run is checked for `Rfact=0.04`.
- **R0 dose at range factor 0.04: done.** `M2-R0-option4-bf`, boundary-free, 20 × 3 × 10⁶ histories: D/Φ₀ = 5.3708 × 10⁻¹² Gy cm² ± 0.14%. Against the M1 boundary-free value (range factor 0.08) the ratio is 0.99990 ± 0.00166. The step size changes the R0 dose by less than 0.17% (1σ). A 25-batch extension (`-ext`) brings the pooled dose to the 0.1% target.
- **R0 layers at range factor 0.04: done.** `M2-R0-option4`, 20 × 6 × 10⁶ histories, is now the layer reference. Against M1, which used range factor 0.08, 54 scalars were compared.
  - No physical quantity moves by more than 1.6σ. The largest are the mean initial electron energy (+0.15% ± 0.11%) and the scattered share of interactions (−0.14% ± 0.11%).
  - Three quantities exceed 2.5σ, none of them for a physical reason. The history-by-history relative SEM scales with the number of histories. The maximum Compton recoil energy is an extreme-value statistic.
  - The third is the positron share of electrons set in motion, at 1.75 × 10⁻⁴ against 2.03 × 10⁻⁴ (−3.3σ). Pair production does not depend on the electron step size. Its batch values are heavy-tailed, from 1.1 to 2.2 × 10⁻⁴ in M2, and M1's main run and replicate already differ by 2.0σ. I read it as a fluctuation in a statistic that the importance weights make heavy-tailed. It enters no dose quantity.
  - The M2 event sample (2 × 10⁵ interactions) gives δ_C = 60 keV (D014), the same as M1.
- **R0 dose reference: done.** `M2-R0-option4-bf` (20 batches) and `-ext` (25 batches, new seeds) agree, z = −0.64, so they are pooled as under D022/D027.
  - D/Φ₀ = 5.3672 × 10⁻¹² Gy cm² ± 0.090%, which meets the 0.1% target.
  - Against the M1 boundary-free value it is 0.99924 ± 0.00125.
  - β = 1.0053 ± 0.0014, with K_col from `M2-R0-option4`; K_col itself is 1.0009 ± 0.0012 times M1.
- **EGSnrc s_w,air (SPRRZnrc, B0): done.** 1.12602 ± 0.00002 (4 jobs × 2.5 × 10⁷). Geant4's R0 electron spectrum through Spencer–Attix with the same matched PEGS4 stopping powers gives 1.12593 ± 0.000003, a difference of +0.008%. The Geant4 value in B0 itself follows from `M2-B0-option4`.
- **PDD(5), boundary-free: done.** `M2-R0-option4-pdd`, cylinder at 8 cm, range factor 0.04: PDD(5) = 81.79 ± 0.22%, d_max = 0.81 ± 0.04 cm (parabola through the 1 mm bins of a flat peak, so d_max is loosely determined). M1 gave 81.53 ± 0.26% with D(5) from the cylinder (−0.24% artifact). The difference is +0.26 ± 0.34 points.
- **Livermore and Penelope at R0: done.** Both use the same msc model and step settings as option4 (GS below 100 MeV, SafetyPlus, range factor 0.04, from their batch-0 printouts). Their differences from option4 therefore come from the cross-section and energy-loss models. Against option4:
  - Photon fluences, mean photon energy, kerma, K_col, g and the dose shares agree within 0.2% (≤ 1.8σ).
  - s_w,air agrees within 0.003% on every stopping-power basis.
  - The scoring-cylinder dose agrees within +0.25% (Livermore, 1.6σ) and +0.11% (Penelope).
  - Resolved model differences:
    - Penelope's electron fluence above 10 keV is −0.80% (−4.9σ) and its mean electron energy −0.16% (−3.0σ). Its ionisation model gives a higher stopping power, so at the same dose there is less fluence.
    - Livermore's scattered share of interactions is +0.48% (+3.9σ) and its mean initial electron energy −0.36% (−2.7σ). Both point to its Compton model (binding and Doppler).
  - The largest list difference in any scalar is 0.8%. The scoring tolerances are 2% and 5%, so the AI comparison can resolve every quantity at those levels.
  - **Distributions.** List-to-option4 distances are of the same order as the option4 half-sample distances. The largest is Penelope's electron angular distribution, with W1 = 0.0014 against 0.0004 for the half sample.
- **EGSnrc dose benchmark in B0: done.**
  - DOSRZnrc (16 jobs × 8 × 10⁷, analog, EXACT boundary crossing, matched media): D/Φ₀ = 5.3723 × 10⁻¹² Gy cm² ± 0.13%. This is the dose per surface fluence × (105/100)².
  - Geant4 option4, boundary-free in B0 (`M2-B0-option4-bf`): 5.3829 × 10⁻¹² ± 0.11%.
  - EGSnrc / Geant4 = 0.99803 ± 0.00169 (−0.20%, −1.2σ). That is inside the plan's 0.5% investigation threshold (Section 4.2), and no investigation is needed.
- **EGSnrc fluence and s_w,air benchmark in B0: done.**
  - Runs: FLURZnrc 8 × 6 × 10⁷, SPRRZnrc 4 × 2.5 × 10⁷, Geant4 `M2-B0-option4` 20 × 3 × 10⁶, same scoring cylinder at 5 cm.
  - EGSnrc / Geant4:
    - s_w,air, matched media: 1.000079 ± 0.000016 (+0.008%).
    - Total photon fluence: 1.0014 ± 0.0007 (2.1σ).
    - First-generation photon fluence: 1.0011 ± 0.0010. EGSnrc counts bremsstrahlung photons as first-generation, Geant4 as secondary.
    - Electron fluence above 10 keV: 1.0046 ± 0.0030 (1.5σ). FLURZnrc spreads each step's track length linearly in energy; Geant4 uses the CSDA-weighted estimator (D020).
  - Every B0 quantity agrees within 0.5% (Section 4.2 threshold), so the benchmark passes.
  - **Electron spectrum shape.** EGSnrc/Geant4 is +5% in the 10–20 keV bin, as D020 predicts for a linear-in-energy track-length estimator. It is within ±1% from 0.1 to 1.0 MeV, with structure of about ±1% near the Compton edges (0.96 and 1.12 MeV), where the bound-Compton models differ. Photon spectra agree bin by bin within the statistics (a few %).
- **Scorer positive controls.** The Livermore and Penelope results, scored as if they were model answers, show no breakpoint. Livermore passes everywhere. Penelope passes everywhere except one unresolved distribution (outside 3 × half-sample, inside the code spread).
- **Queue** (`mc/queues/M2.yaml`, run by `mc/tools/run_queue.py` in chunks under 2 h):
  - option4 layers, then the dose extension;
  - SPRRZnrc, the boundary-free PDD;
  - Livermore and Penelope;
  - DOSRZnrc, Geant4 B0 boundary-free, FLURZnrc, Geant4 B0 layers;
  - Livermore and Penelope boundary-free.
- **Binary check.** The M2 binary (`co60app.m2`, adds geometry B0) reproduces the frozen production binary bit for bit at R0: same seeds, 10⁵ histories, every tally identical. R0 runs use the production binary and B0 runs the M2 binary.

### What failed or was anomalous
- **Normalisation slip in my new M2 dose code, caught before use.**
  - What went wrong: the boundary-free runs place the scoring cylinder at 8 cm, and co60app reports Φ₀ at the cylinder's depth (108 cm from the source). My first `analysis/m2.py` divided by that Φ₀ and put D/Φ₀ 5.8% high, which is (108/105)².
  - How it was caught: the comparison with M1, whose analysis normalises to 105 cm.
  - Fix: `phi0_at_r0()` now recomputes Φ₀ at 105 cm from the solid angle. It is checked against the main run, where both are the same.
- **Queue runner shadowed Python's `queue` module.** I named it `mc/tools/queue.py`. When the queue reached the first EGSnrc run, `run_egsnrc.py` imported `concurrent.futures`, which imports the standard `queue` module, and found mine instead. The runner crashed at 18:21 before any job started, and the empty run directory was reused on relaunch. Renamed to `run_queue.py`. About 20 min lost.
- **Comparison runs trimmed before they started.** Livermore and Penelope at R0 use 4 × 10⁶ histories per batch, and Geant4 B0 layers 3 × 10⁶, to save about 3 h. The option4 reference keeps 1.2 × 10⁸.
- **The s_w,air check failed in its first form (2026-10-29, D039).**
  - Applied to EGSnrc's fluence with the two-sided track-end reading used for Geant4, our calculation came out 0.022% below SPRRZnrc (11σ).
  - The cause was the reading, not the calculation: EGSnrc has no fluence below its 10 keV cut-off, so averaging the bins on both sides of Δ halved the track-end term.
- **FLURZnrc parser window (2026-10-29).** `flurz_spectra` read only 20,000 characters after each region header. That holds the 134 rows of a 10 keV spectrum but would have cut a 667-row 2 keV spectrum to about 250 rows without an error. It was found by reading the code before the 2 keV run was analysed, and it now reads to the end of the block. A check that the parsed spectrum reaches 1.3325 MeV guards it.
- **I misread the B0 electron spectra once (2026-10-29).** I first wrote that Geant4's CSDA-weighted estimator matches FLURZnrc at low energy. The per-bin comparison shows a 6% difference at 10–20 keV, and the STATUS text was corrected before it was committed.

## M3. AI harness (2026-10-11). Built and tested offline; the P0 pilot and M4 need keys, network access and budget approval.

### Summary

The harness is complete and tested without any API call. It covers prompts, schemas, the sandbox, the provider clients, the run matrix, logging and the cost estimate. The M5 scorer is built and validated on controls. The pilot (P0) and the full matrix (M4) cannot run here:
- there are no API keys in the environment;
- the network policy blocks `api.openai.com`, `openrouter.ai`, `api.together.xyz` and `physics.nist.gov`;
- paid calls need Humza's approval of a cost estimate (Section 10.5).

`ai/README.md` has the launch checklist.

### What was done

- **Observables** (`ai/observables.yaml`). 33 quantities across the five layers, in the units and on the bins of `config/bins.yaml`, plus the sampled-events fields. Each maps to its Monte Carlo key, a tolerance class (D028) and hypothesis tags (H1 primary photon, H2 transport, H3 recall/compute). It is the single source for the prompts, the JSON schema and the scorer.
- **Prompts** (`ai/prompts/v1/`). Built by `ai/harness/build_prompts.py` and hashed in `MANIFEST.yaml`:
  - system prompts for A0, A1 and A2;
  - observables and events tasks for R0 and P0;
  - the two JSON schemas.

  `tests/test_ai_prompts.py` fails if any number in a prompt, other than the configuration and the bin edges, lies within 0.5% of any Monte Carlo result or `refs.yaml` value (Section 10.2).
- **Sandbox** (`ai/harness/sandbox.py`). Model code runs in new network, mount and PID namespaces:
  - no internet;
  - the repository, Geant4, EGSnrc, their builds and sources, the queue logs and the session transcripts are hidden;
  - no other process is visible, and the environment is empty;
  - limits: 4 cores, 10 min per call, 16 GB address space.

  Tested: DNS fails, every hidden path is empty, and the A2 data appear at `/tmp/data`.
- **Clients and runner** (`ai/harness/providers.py`, `run.py`):
  - Anthropic Messages, OpenAI Chat Completions, Gemini generateContent, and OpenAI-compatible for open-weight hosts, over HTTPS with the standard library;
  - one tool loop, capped at 30 tool calls;
  - one logged retry on malformed JSON;
  - API errors recorded, never retried silently beyond HTTP backoff;
  - every request, response, tool call and outcome written once, read-only, to `ai/raw/`.

  The runner refuses R0 runs until the model list and prompts are frozen. A mock provider tests all of this offline (8 tests).
- **A2 data** (`ai/data/`). NIST was unreachable from both the container and the fetch tool, so the XCOM and ESTAR water tables are reproductions:
  - built from the NIST data that EGSnrc distributes: XCOM elemental cross sections, ESTAR density-effect corrections and the Seltzer–Berger bremsstrahlung tables;
  - computed by XCOM's and ESTAR's methods;
  - labelled as reproductions in their headers and in `refs.yaml`.

  Below 1.4 MeV they agree with Geant4 option4 within 0.1% (photon coefficients above 30 keV, radiative stopping power) and within 0.6% (collision stopping power above 0.5 MeV, I = 75 vs 78 eV).
- **Cost estimate** (`ai/prices.yaml`, `run.py --dry-run`):
  - **Matrix:** 125 conversations, for 5 models × (3 conditions × k = 5 observables + 2 conditions × 5 event calls).
  - **Estimate:** about $130 with central list-price assumptions; the high case (3× prices) is about $400. The P0 pilot is about $10 per model.
  - **Basis:** the prices come from third-party October 2026 tables that disagree with each other, and the token counts are assumptions. Both are replaced by official prices and P0-measured usage before approval.

### What failed or was anomalous

- **NIST unreachable.** physics.nist.gov is blocked by the container's network policy. A fetch agent's separate route could not resolve the name either. No value was transcribed from memory.
- **First ESTAR reproduction was wrong by up to 2× in the radiative stopping power.** I read the Seltzer–Berger table in the wrong order: it is stored k/T-major, 57 energies per k/T. The validator against Geant4 caught it before anything used the file. It is fixed and documented in the builder.
- **Model identifiers are not in the repository.** `ai/models.yaml` names slots and the selection rule (D028); exact IDs are entered at launch and recorded per call.

### Next

The P0 pilot and M4 follow the checklist in `ai/README.md`. Then the M5 scorer runs on the answers.

## M1. Monte Carlo reference at R0 (2026-10-10). Reviewed at the gate; decisions D024–D027. Complete.

### Summary

The Geant4 reference for all five layers at R0 is done: G4EmStandardPhysics_option4, 10 keV thresholds, 2.4 × 10⁸ histories in 20 batches (main run and a seed-independent replicate, pooled by the rule fixed in advance, D022). Every statistical target in plan Section 4.1 is met.

- **Dose.** D/Φ₀ = 5.3713 × 10⁻¹² Gy cm² ± 0.087%. This is the boundary-free value, adopted under D025/D027; the scoring cylinder gives 5.3584 × 10⁻¹² ± 0.066%, 0.24% lower.
- **β.** β = D/K_col = 1.0069 ± 0.0011 with the boundary-free D (1.0045 ± 0.0006 with the cylinder D).
- **s_w,air.** Scoring reference (D024): ICRU 90 basis, with ICRU 37 alongside. The Spencer–Attix ratio (Δ = 10 keV) is 1.1331 on the ICRU 37 stopping-power basis, 1.1268 on ICRU 90 and 1.1385 on Geant4's own stopping powers. The 1% spread between bases is far larger than the statistics.

Problems found and fixed or bounded:
- **Two Geant4 problems that would have biased results (fixed).** Rayleigh scattering of primaries is missing in option4's general gamma process (D019). The plan's linear track-length estimator gives the wrong electron fluence at low energy (D020).
- **10-batch standard errors can come out much too small by chance (D023).** This made a primary-fluence fluctuation in the main run look like a 4σ effect; the replicate resolved it.
- **The scoring cylinder's boundary distorts the dose by up to ±3% in the few millimetres next to it and lowers it by 0.24% inside.** After the 20-batch extension, the cylinder dose is 0.99759 ± 0.00109 of the boundary-free dose (−2.2σ). By the rule in D025, with the threshold set in D027, the boundary-free value is now the R0 dose reference. M2 re-measures it with boundary-free scoring.

Results page: `analysis/results/co60_r0_m1.html`, published (private) at https://claude.ai/artifact/Cu5ScfiTccG3u14D4mbiLe. Study brief (why, what, results, strengths and weaknesses, venues): `analysis/results/co60_brief.html` (built by `analysis/m1_brief.py`), published (private) at https://claude.ai/artifact/24vAALYwGGg9Hej4P7Df4d.

### What was done

**Application** (`mc/geant4/co60app/`). A Geant4 11.4.3 application scoring the five layers in the R0 geometry:
- point source at SSD 100 cm, uniform in solid angle over the 10 × 10 cm² pyramid, both lines with equal probability;
- 30 cm water cube, scoring cylinder r = 1 cm, t = 0.2 cm centred at 5 cm depth;
- production thresholds set by energy (10 keV for e−/e+ in G4_WATER, range 8.25 µm);
- every photon carries its history: primary per line, scattered source photon, or secondary photon. Every electron carries its parent photon's category and whether it is a delta ray;
- tallies in MeV, cm and g; JSON output per batch; per-history second moments for D, K and the total photon track length.

**Variance reduction** (D016). Electron Russian roulette outside a region of interest, and photon importance sampling in a parallel world (importance 2, 8, 32). The efficiency gain for D is about 25× over analog transport. An analog run checks it (below).

**Runs** (`mc/runs/`, each with a manifest):

| Run | Purpose | Histories |
|---|---|---|
| M1-R0-option4, M1-R0-option4-rep2 | main run and replicate, pooled (D022) | 2 × 10 × 1.2 × 10⁷ |
| M1-R0-option4-S2-5keV | check S2: 5 keV thresholds | 10 × 6 × 10⁶ |
| M1-R0-option4-S1-half | check S1: r 0.5 cm, t 0.1 cm | 10 × 9 × 10⁶ |
| M1-R0-option4-pdd | central-axis depth dose, PDD(5) | 10 × 6 × 10⁶ |
| M1-vrcheck-analog | no variance reduction | 10 × 6 × 10⁶ |
| M1-thinslab-option4-v2 | thin-slab track-length test | 10 × 4 × 10⁵ |
| M1-check-transmission | tracking μ by transmission (TestEm13) | 2 × 2 × 10⁷ photons |
| M1-check-primary-* | primary-fluence diagnostics (5 runs) | see below |
| M1-check-pdd-cyl8 | depth scan with the cylinder at 8 cm | 4 × 3 × 10⁶ |
| M1-check-boundary-free-R0, -ext | boundary-free dose at R0 (main VR, cylinder at 8 cm) | 10 × 6 × 10⁶ + 20 × 6 × 10⁶ |
| M1-tables-option4-v2 | stopping powers, δ, cross sections from the production physics | — |

Compute ran in this 4-core container (D013). Wall times per batch are in each manifest; runs often shared the CPU, so they overstate CPU use.

### Results at R0 (pooled, 20 batches)

Uncertainties are 1σ standard errors of the mean: the larger of the batch spread and the history-by-history estimate where both exist (D023).

| Layer | Quantity | Value |
|---|---|---|
| 1 | total photon fluence / Φ₀ | 1.1541 ± 0.0004 |
| 1 | scattered fraction of fluence / of energy fluence | 0.3685 / 0.1810 |
| 1 | mean photon energy, all / scattered | 0.9666 / 0.4747 MeV |
| 2 | interactions: Compton, photoelectric, Rayleigh, pair | 96.52%, 2.43%, 1.03%, 0.020% |
| 2 | share of energy transferred by Compton | 99.77% |
| 3 | mean initial electron energy: all / primary parent / scattered parent | 0.3472 / 0.5887 / 0.1263 MeV |
| 3 | Compton recoils above the free-electron edge (1.1732 / 1.3325 MeV) | 0.22% / 0.18% |
| 4 | delta-ray share of electron fluence above 10 keV | 0.0300 |
| 4 | radiative / collision energy loss | 0.00229 |
| 5 | D/Φ₀, R0 reference (boundary-free, D027) | 5.3713 × 10⁻¹² Gy cm² ± 0.087% |
| 5 | D/Φ₀ in the scoring cylinder | 5.35838 × 10⁻¹² Gy cm² ± 0.066% |
| 5 | K/Φ₀, K_col/Φ₀ | 5.3483, 5.3343 × 10⁻¹² Gy cm² |
| 5 | β = D/K_col, boundary-free D (cylinder D) | 1.0069 ± 0.0011 (1.0045 ± 0.0006) |
| 5 | g | 0.00262 |
| 5 | dose from electrons set in motion by scattered photons | 18.8% |
| 5 | s_w,air (Spencer–Attix, Δ = 10 keV): ICRU 37 / ICRU 90 / Geant4 / PEGS4 matched | 1.1331 / 1.1268 / 1.1385 / 1.1259 |
| 5 | PDD(5), d_max | 81.53 ± 0.26%, 0.78 ± 0.05 cm |

The ICRU 37-basis s_w,air, 1.1331, agrees with the TRS-398 value in refs.yaml (1.133, not yet verified). PDD(5) has no external value yet; refs.yaml `pdd5` is null. The depth of maximum dose is poorly defined, because the dose is flat to within about 0.5% between 0.45 and 0.95 cm.

### Checks

| Check | Result | Status |
|---|---|---|
| Statistical targets (Section 4.1) | D/Φ₀ 0.066% (target 0.1%), β 0.062% (0.2%), worst spectral bin 0.57% (1%) | met |
| Thin slab: track-length fluence against analytic, no VR | 0.9995 ± 0.0007 | passes |
| Tracking μ against G4EmCalculator (TestEm13 transmission) | +0.10 ± 0.04% (1.1732 MeV), +0.07 ± 0.04% (1.3325 MeV) | measured; the analytic fluence uses tracking μ |
| Primary fluence at R0 against analytic (tracking μ) | pooled 0.9992 ± 0.0004 (−1.7σ); main 0.9988 ± 0.0006, replicate 0.9996 ± 0.0006 | passes |
| Fluence-to-dose closure (CSDA-weighted estimator) | 1.0007; linear estimator 1.0644 (D020) | passes |
| S2: 5 keV thresholds | D/Φ₀ +0.13% (+0.9σ); s_w,air +0.0003%; electron fluence normalization +0.12%, shape χ² 20/24; photon fluence normalization −0.03%, shape χ² 29/26 | passes |
| S1: half-size scoring volume | D/Φ₀ +0.47% (+2.1σ); electron fluence normalization +0.42%, shape χ² 23/24; s_w,air +0.0010% | passes (under 3σ) |
| Analog transport against VR | D/Φ₀ −0.10% (−0.2σ); electron fluence normalization −0.04%, shape χ² 8/22 | passes |
| Scoring-cylinder boundary: D in the cylinder over the boundary-free D at R0 | 0.99759 ± 0.00109 (−2.2σ; boundary-free D from 30 + 4 batches) | boundary effect resolved; boundary-free D adopted (D027) |
| Event sample, validity rules (D014) | 0.064% of 100,000 Monte Carlo interactions fail the bound rules; 99% fail the strict free-electron rules | as designed |
| Bookkeeping | 0 untagged tracks; largest per-interaction energy imbalance 1.23 keV | passes |
| Fano step-size diagnostics | msc range factor 0.04: 1.00001 ± 0.00088; step function 0.05 / 1 µm: 0.99921 ± 0.00073 (20 batches each; production 1.00157 ± 0.00058) | deviation gone with either tightening; M2 uses range factor 0.04 |

### What failed or was anomalous

- **Rayleigh scattering missing for primaries (fixed, D019).** In option4, G4GammaGeneralProcess does not sample Rayleigh scattering above 1.022 MeV, so primary photons could never scatter coherently. It showed up in the first production attempt as zero Rayleigh events among primaries. That attempt was stopped and archived (`mc/runs/_archived/M1-R0-option4-attempt1/`). All production runs switch the general process off.
- **Plan's linear electron-fluence estimator fails closure (D020).** It puts too much track length at low energy for the final steps of electrons below about 20 keV, so the dose rebuilt from it is 6.4% too high. A CSDA-weighted estimator closes to 0.07% and is the Layer 4 reference. The linear one is kept and reported; on s_w,air the two differ by 0.008%.
- **Photon categories wrong at importance-cell boundaries (fixed before production).** Split copies were labelled as secondary photons, and primaries became "scattered" when crossing a cell boundary. Interactions are now counted only for electromagnetic processes, and split copies inherit the parent's history. Every production run reports 0 untagged tracks.
- **The scoring cylinder's boundary distorts the dose next to it.** The PDD run's 1 mm depth slabs (r < 1 cm) show the dose 2–3.7% high 1.5–4.5 mm in front of the scoring cylinder and 1.5–2.3% low 1.5–4.5 mm behind it. Every batch shows it (11σ), with water on both sides of the boundary.
  - Moving the cylinder to 8 cm (`M1-check-pdd-cyl8`) moves the pattern with it.
  - Far from both cylinders the two depth scans agree to 1.0011 ± 0.0008 per history.
  - Inside the cylinder: against the boundary-free dose at R0 (`M1-check-boundary-free-R0`: the main run's variance reduction, the cylinder at 8 cm, the dose scored by position in r < 1 cm, 4.9–5.1 cm) the ratio is 0.99759 ± 0.00109 (−2.2σ; 95% interval −0.46% to −0.02%) after the 20-batch extension (30 batches). The analog depth scan `M1-check-pdd-cyl8` (4 batches) is combined with it by inverse variance and carries about 1% of the weight. The boundary-free value has batch errors only. Under D025/D027 it is now the R0 dose reference.
  - Likely mechanism: with GS multiple scattering and the SafetyPlus step limit, electron steps shrink as electrons approach a boundary (geometry factor 2.5, range factor 0.08), while far from boundaries they reach 0.2 × the range. A step-length-dependent displacement of the energy deposition piles energy up in front of the boundary and leaves a deficit behind it. Energy is moved, not created.
  - PDD(5) uses the dose inside the cylinder and D_max far from it.
- **Primary-fluence fluctuation in the main run (resolved).** Against the analytic exp(−μs)/(ΩR²) with the tracking μ measured by TestEm13, the main run gave 0.9988. That is −4.4σ on its batch spread and −1.9σ on counting statistics. The diagnostic runs found no cause:

  | Run | Transport | VR | Scoring cylinder | Primary / analytic |
  |---|---|---|---|---|
  | main | full | importance + roulette | r 1, t 0.2 | 0.9988 ± 0.0006 |
  | replicate | full | importance + roulette | r 1, t 0.2 | 0.9996 ± 0.0006 |
  | S2 | full | importance + roulette | r 1, t 0.2 | 0.9996 ± 0.0010 |
  | analog | full | none | r 1, t 0.2 | 0.9989 ± 0.0010 |
  | is-on-prodgeom | photon only | importance | r 1, t 0.2 | 1.0005 ± 0.0011 |
  | is-mass-prodgeom | photon only | importance, mass-geometry cells (D021) | r 1, t 0.2 | 0.9987 ± 0.0012 |
  | is-off | photon only | none | r 4, t 1 | 1.0005 ± 0.0004 |
  | is-on | photon only | importance | r 4, t 1 | 0.9998 ± 0.0004 |
  | full-r4 | full | none | r 4, t 1 | 1.0007 ± 0.0004 |
  | thin slab | full | none | r 4, t 0.05 at 0.5 cm | 0.9995 ± 0.0007 |

  "Photon only" stops electrons in their first step (`/process/em/lowestElectronEnergy 2 MeV`); TestEm13 shows this parameter leaves photon transmission unchanged (identical counts with identical seeds). Also ruled out: null interactions (none among 10⁵ sampled interactions), history counting, and a scoring artifact (the collision estimator shows the same value).

  The main run's batches agree with each other better than their history variance allows (D023). The replicate, run under the rule fixed beforehand (D022), gave 0.9996. Pooled: 0.9992 ± 0.0004.
- **Batch standard errors can be too small (D023).** The main run's batch spread was 0.43–0.71 of the history-based value. Targets and comparisons use the larger of the batch and history-based errors. Spectra have only batch errors.
- **Spectrum comparisons were not shape comparisons.** The first χ² between run spectra compared absolute fluences per Φ₀, so S1's overall normalization difference showed up as a large χ². The analysis now reports the normalization ratio and a shape χ² (each spectrum scaled to unit sum) separately.
- **Geant4's water stopping power differs from the Bethe formula at low energy.** In option4, water's collision stopping power is above Bethe–Møller (I = 78 eV, same density effect) by 2.5% at 12 keV, 1.7% at 100 keV and 0.9% at 200 keV, and within 0.4% above 0.5 MeV. Air agrees within 0.1%.
  - Option4 uses the Penelope ionisation model below 100 keV (`PenIoni`, then `MollerBhabha`).
  - Geant4 smooths the dE/dx table across the model boundary, carrying the 1.7% offset at 100 keV as 1.7% × (100 keV/E) above it. That accounts for the 0.85% at 200 keV and 0.34% at 500 keV.
  - So the Geant4-basis s_w,air (1.1385) is 1.0% above the ICRU 90 basis (1.1268). PEGS4 media built with Geant4's I, density, composition and density effect give 1.1259 (`mc/egsnrc/b0/`, D015).
- **Two slips in my own process management.** Twice a `pkill -f` pattern matched my own shell, and once a waiting script as well. No simulation ran with a wrong configuration. The only run lost was a 2-minute-old extension, archived as `mc/runs/_archived/M1-check-pdd-cyl8-ext-stopped/` with its reason. Processes are now stopped by PID only.
- **The ROI check has no buffer on the near side of a depth scan.** `RunAction::CheckRoi` requires the ROI to start at or before the start of the depth scan, but not 1.2 × the CSDA range before it. That is harmless for scans from the surface (the PDD run) but would allow roulette bias for a scan starting inside the phantom. `M1-check-boundary-free-R0` satisfies the buffer by its spec (ROI from 4.1 cm, scan from 4.9 cm). Fixed in the source for M2 (D025); the frozen M1 production binary is unchanged, and every M1 spec satisfies the stricter check.
- **Container restart, 2026-10-10 ~22:55 UTC.** The cloud container was reclaimed while no task of mine was attached to the session, and the three running jobs stopped mid-batch: the boundary-free extension at 2/20 batches, the mass-importance check at 5/8, and Fano range factor 0.04 at 6/20. No finished batch was lost. All three were resumed at 02:11 UTC with the same seeds, so the results are identical to an uninterrupted run. `run_fano.py` gained `--resume` for this; it keeps the interrupted batch directory, renamed `*.interrupted-<time>`. About 3 h of compute was lost. From now on a background task stays attached while jobs run.
- **Second container restart, 2026-10-11 ~06:50 UTC.** The keep-alive task reached the 2 h limit on background tasks at 06:46 and the container was reclaimed within minutes, during batch 15 of the Fano step-function run (batch 15 had started at 06:44 and normally takes about 9 min). Batches 0–14 were kept. The run was resumed at 13:16 UTC with the same seeds (`run_fano.py --resume`). About 6.5 h of wall time was lost, of which 9 min was compute. Lesson: a waiter cannot outlast the 2 h limit, so a job longer than that must run as the attached task itself, in pieces under 2 h each, or be resumed when the session next wakes.
- **Mass-geometry importance check (D021), finished 2026-10-11.** Photon only, production cells and scoring cylinder: primary/analytic 0.9987 ± 0.0012 (−1.1σ). The parallel-world run with the same geometry gives 1.0005 ± 0.0011; the two differ by −0.0018 ± 0.0016 (−1.1σ). Neither implementation shows a bias at the 0.1% level. Production keeps the parallel world (D016).
- **Fano step-function diagnostic finished (2026-10-11 14:45 UTC).** Step function 0.05 / 1 µm (production 0.2 / 10 µm): 0.99921 ± 0.00073 (−1.1σ). Both tightened settings remove the production deviation, so it is a step-size effect of the electron transport. M2 runs at range factor 0.04 (D026, D028).
- **Fano step-size diagnostics.** They were starved of CPU behind production, with one batch taking 4 h, and later resumed at normal priority. Range factor 0.04 gives 1.00001 ± 0.00088, so the production deviation (+0.16%) disappears with tighter msc stepping. The difference is 1.5σ. See D026.
- **Importance-ratio warning in S1.** S1's inner cell has importance 48 against 8 outside, a ratio of 6. Geant4 warns once (GeomBias1001) when the ratio is outside [0.25, 4]. The algorithm handles any ratio fairly: split into ⌊6⌋ = 6 copies of weight w/6. The warning has no effect on results.

### Decisions at the gate (2026-10-10)

Humza approved all recommendations:

1. **s_w,air scoring reference (D024).** ICRU 90 basis primary, ICRU 37 reported alongside; the Geant4 basis is not used for scoring.
2. **Scoring-volume boundary (D025, D027).** The extension resolved the effect: the cylinder is 0.24% low (−2.2σ), so the boundary-free D is now the R0 reference. The 2σ threshold is recorded in D027 for confirmation. From M2, dose at R0 is scored by position, with no geometry boundary within an electron range.
3. **Uncertainty policy (D023).** Approved.
4. **Layer 4 estimator (D020).** CSDA-weighted is the reference; approved.
5. **Fano (D026).** +0.16% accepted as a bound; step-size diagnostics are reviewed before M2 production. M2 runs use at least 20 batches.
6. **refs.yaml (D026).** Humza verifies, in order:
   1. line energies;
   2. TRS-398 s_w,air (and whether its 2024 revision uses ICRU 90);
   3. I-values;
   4. normalisation constants;
   5. the rest.

   A published PDD(5) is to be added.

### Next, after approval (M2)

The B0 code-to-code benchmark: EGSnrc against Geant4 at R0 with the matched water (D015), and Livermore and Penelope runs at R0.

## M0. Environment (2026-10-10). Waiting for Humza's review.

### Summary

Geant4 11.4.3 (multithreaded) and EGSnrc v2026 are built from pinned commits. The Geant4 build reproduces the shipped reference outputs of three examples. All three production EM lists initialize with the installed data. With the production electron-transport settings the Fano cavity test gives 1.0016 ± 0.0006, a +0.16% deviation at 2.7σ. That is a small but statistically resolved failure at the 0.1% level. The negative control fails by 1.5%. Whether the +0.16% matters for R0, and what to do about it, is gate question 2.

Everything ran in a 4-core cloud container, not on the MacBook (DECISIONS D008).

### What was done

**Geant4 11.4.3.** Built from tag `v11.4.3` (commit `2ee379e6`, released 2026-10-01) with gcc 13.3: Release, multithreaded, C++17. Build time: 21 min wall, 70 CPU-min. There were 10 compiler warnings, all `fscanf` unused-result warnings in two Geant4-DNA LEPTS files that this study does not use. Logs: `env/logs/2026-10-10-cloud/`.

**Geant4 data.** The versions are exactly those 11.4.3 requires; G4EMLOW is 8.8. They came from conda-forge packages of the CERN tarballs, because the CERN data server is blocked from the container (D004). Hashes are in `env/versions.lock.yaml`.

**EGSnrc v2026.** Commit `bd49a47a`, configuration `linux`, gfortran 13.3. tutor1, DOSRZnrc, FLURZnrc and SPRRZnrc compile without warnings. Both smoke tests finished: tutor1, and the DOSRZnrc template, a 1.25 MeV depth dose in water with 100 histories (`mc/runs/M0-egsnrc-smoke/`).

**Geant4 example checks** (`mc/runs/M0-g4-examples/`). Each example ran its shipped macro, and the result was compared with the output the Geant4 team ships with it.

| Example | Check | Result |
|---|---|---|
| TestEm0 (deterministic: cross sections, stopping powers, ranges) | 32 result lines against the shipped output | identical |
| B1, 6 MeV photons | dose against the shipped output | z = −0.36 |
| B1, 210 MeV protons (exercises hadronic data) | dose against the shipped output | z = +1.04 |
| fanoCavity, shipped macro (standard_opt3, 4000 events) | ratio and mean secondary-electron energy | z = +0.43 and −0.96; Compton cross section identical |

z is the difference divided by the combined uncertainty (`comparison.yaml`).

**Physics-list initialization** (`mc/runs/M0-physics-init/`). G4EmStandardPhysics_option4, G4EmLivermorePhysics and G4EmPenelopePhysics each initialize in G4_WATER with the installed data and compute cross sections at 1.1732 and 1.3325 MeV (photons) and 1 MeV (electrons). There were no G4Exceptions and no warnings. The runtime material printout gives Imean = 78.0 eV for G4_WATER, which matches the source code (`refs.yaml`: `water_mean_excitation_energy_geant4`). The printed cross sections are archived and not interpreted here.

**Fano cavity test.** Configuration and coverage are in DECISIONS D006 and D007. The setup is 1.25 MeV photons, a G4_WATER wall 5 mm thick on each side, a 2 mm × 1 cm radius cavity of G4_WATER at 1 mg/cm³, no delta rays and no bremsstrahlung. The uncertainty comes from the spread of the batch results; the example's own event-by-event estimate agrees with it.

| Configuration | Runs | Batches × events | (D/Ψ)/(μ_tr/ρ) | Deviation from 1 | 95% interval |
|---|---|---|---|---|---|
| standard_opt4 (production e- transport) | `M0-fano-opt4`, `M0-fano-opt4-ext` | 40 × 340 000 | 1.00157 ± 0.00058 | +0.16% (2.7σ) | 1.0004 to 1.0027 |
| standard_opt0 (negative control) | `M0-fano-opt0-control` | 10 × 100 000 | 1.0147 ± 0.0016 | +1.47% (9.4σ) | 1.0116 to 1.0178 |

The two standard_opt4 runs agree with each other: 1.0016 ± 0.0011 (10 batches) and 1.0016 ± 0.0007 (30 batches, different seeds). The batch-spread uncertainty (0.058%) agrees with the example's own estimate (0.053%). Pooled numbers are in `analysis/M0_fano_summary.yaml`.

Interpretation, for discussion at the gate:
- The deviation is about one tenth of the opt0 failure. It is below the 0.5% investigation threshold for the code-to-code benchmark but above the 0.1% statistical target for D/Φ₀.
- The test cavity is water vapour at 1/1000 of the wall density, the hardest case for boundary crossing. In R0 the scoring volume is water inside water, with a geometric boundary but no change of density, so this result bounds interface artifacts for a 1000:1 density step. It does not translate directly into a bias on R0 dose.
- Not yet tested: whether the deviation shrinks with tighter stepping. A drop would identify it as a step-size artifact; no change would point elsewhere, for example to the example's own approximations.

![Fano test batches](analysis/figures/M0_fano.png)

Throughput in the container for standard_opt4 with 4 threads was 340 000 events per 191 s, about 1.8 × 10³ events/s. This number is specific to the Fano geometry and its 1000× Compton cross-section factor, so it is not a production estimate.

**Repository.**
- Layout follows plan Section 8.
- `refs/refs.yaml` lists 15 external values, all `verified_by_humza: false`. Five are marked `recall`, typed from memory because their sources are not reachable from the container; check those first. NIST coefficient values are not entered yet.
- 12 unit tests pass: Compton kinematics, including the Layer 3 edges of 0.9634 and 1.1181 MeV, and the refs.yaml schema.
- Every run has a manifest in `mc/runs/` with git commit, code and data versions, physics, cuts, seeds, histories, wall time and machine.

### What failed or was anomalous

- **CERN unreachable.** `geant4-data.web.cern.ch`, `gitlab.cern.ch` and `physics.nist.gov` are blocked by the container's network policy. Geant4 source came from the GitHub mirror and data from conda-forge (D002, D004). NIST values could not be fetched.
- **dosxyz_show failed to compile** during EGSnrc configuration. It is the X11 viewer for DOSXYZnrc and is not used.
- **DOSRZnrc prints `***ERROR*** ... NOT FOUND`** for optional transport-parameter inputs that are absent. This is standard EGSnrc behaviour: defaults are used and the run completes.
- **Manifest marked dirty.** The EGSnrc smoke-test manifest records `git dirty: true`, because it ran before the scaffold was first committed. The run is kept as is.
- **No data environment variables.** In 11.4, `geant4.sh` does not export the data environment variables; the data path is compiled in. `geant4-config --datasets` also lists datasets that are not installed (G4NDL, RealSurface, G4CHANNELING). Neither affects EM physics. Manifests record the dataset paths that geant4-config reports.
- **Fano deviation resolved by extension.** The first standard_opt4 Fano run (10 batches) gave 1.0016 ± 0.0011, at 1.4σ. That precision could not distinguish a 0.1% artifact from zero, so 30 more batches were run (`M0-fano-opt4-ext`, DECISIONS D011). Pooled, the +0.16% deviation is resolved at 2.7σ. This is a failure of the Fano test at the 0.1% level and is reported as such.

### Open questions for Humza at this gate

1. **Environment.** Accept Geant4 11.4.3 and EGSnrc v2026 as pinned, and the conda-forge route for the data in the container (D004). On the MacBook, does geant4-data.web.cern.ch resolve, so that the install script can use the CERN tarballs?
2. **Fano result: +0.16 ± 0.06%.** The plan sets no pass criterion. Options:
   - (a) Accept, and document +0.16% as a bound on interface artifacts.
   - (b) Run a step-size sensitivity diagnostic first, for example the msc range factor halved to 0.04, and separately a smaller final range in the step function. Matching the present ±0.06% needs 13.6 × 10⁶ events per setting: 2.1 h at the current settings in this container, and longer with tighter stepping.
   - (c) Treat it as a failure and change the production step settings.

   Recommendation: (b), run alongside M1 application development, with the decision made before any M1 production run. The development work does not depend on the outcome. Separately: the test covers the electron transport shared by the three lists, not their different ionisation models below 100 keV (D006). Accept that coverage, or ask for Fano-modified Livermore and Penelope ionisation models?
3. **Compute for M1** (D008). Run on the MacBook, or in this kind of container? M1 CPU estimates need throughput measured with the study application on the chosen machine.
4. **Electron comparison grid (Section 3).**
   - 10 keV to 1.20 MeV is not a whole number of 50 keV bins. Proposal: edges 0.010, 0.050, 0.100, …, 1.200 MeV, so the first bin is 40 keV wide.
   - Photoelectrons from the 1.3325 MeV line (about 1.33 MeV) lie above the grid. Proposal: add one overflow bin so nothing is silently dropped.
5. **Compton validity checks (Layer 3, Section 6).** All three production lists use bound-electron Compton models with Doppler broadening below 20 MeV: G4LowEPComptonModel in option4, G4LivermoreComptonModel in Livermore, G4PenelopeComptonModel in Penelope. Monte Carlo recoil-electron energies are therefore smeared around the free-electron Compton edge (binding energy subtracted, Doppler broadening of the scattered photon), and the free-electron check will not hold exactly. The check needs either a tolerance or a definition based on bound-electron kinematics. The same applies to the per-event validity checks on AI events.
6. **Water definition for the B0 benchmark (M2) and open decision 5.**
   - EGSnrc's standard `521icru.pegs4dat` water uses ICRU 37 data, with I = 75 eV per refs.yaml `recall`.
   - Geant4 G4_WATER has I = 78 eV.
   - EGSnrc v2026 ships an ICRU 90 water density-correction file, but it uses ρ = 0.998 g/cm³ against G4_WATER's 1.000.

   An I-value mismatch could produce a difference in s_w,air near the 0.5% investigation threshold. The two codes need a matched water definition before M2.
7. **refs.yaml.** Five `recall` entries need checking. NIST μ/ρ and μ_en/ρ for water need entering from a machine that can reach NIST, along with which table to use for μ_tr/ρ.

### Next, after approval (M1)

The study's own Geant4 application for R0, with all five layers scored and checks S1 and S2, the analytic primary-fluence check, and the statistical targets. The first step is a throughput measurement and CPU-hour estimate on the chosen machine.
