# Study plan: photon-to-dose physics at the Co-60 reference point, AI models versus Monte Carlo

Owner: Humza Nusrat (PI)
Status: draft v0.1, October 2026
Audience: Claude Code, building and running the study under Humza's review

## 0. How to use this document

This is the specification. Work milestone by milestone (Section 9) and stop at every review gate. Log every design choice in `DECISIONS.md` and progress in `STATUS.md`. Ask before spending money on API calls or cloud compute. Section 10 lists the rules that apply throughout. Section 11 lists decisions Humza still has to make; where a default is given here, use it until he changes it.

## 1. Question

Language models can state textbook radiation physics. Stating a cross section is different from modelling what photons and electrons do at a point inside a phantom. This study follows the energy from the incident photon to the absorbed dose at one well-characterized point and compares AI-generated answers with Monte Carlo at each step.

The point is on the central axis at 5 cm depth in water in a Co-60 beam with a 10 × 10 cm² field. Standards laboratories use this reference condition to calibrate ionization chambers in absorbed dose to water (N_D,w), so it is one of the best-characterized points in radiation dosimetry.

**Primary question.** At the Co-60 reference point, do AI models describe the correct photon and electron physics, and at which step of the photon-to-dose chain do their answers first depart from Monte Carlo?

**Secondary question.** How closely do independent Monte Carlo implementations (three Geant4 electromagnetic physics lists and EGSnrc) agree at the same point? This spread sets the resolution of the AI comparison. An AI difference smaller than the Monte Carlo spread cannot be interpreted.

**Hypotheses, recorded before any data collection.**

- H1. AI errors are small for primary-photon quantities, which follow from attenuation coefficients and two-body kinematics.
- H2. AI errors are larger for the scattered photon spectrum and the electron fluence spectrum, which require transport.
- H3. AI models reproduce a tabulated recall quantity (percent depth dose at 5 cm) more accurately than a computed quantity at the same point (dose per unit fluence).

## 2. Reference configuration R0

| Parameter | R0 value | Notes |
|---|---|---|
| Phantom | Liquid water, 30 × 30 × 30 cm³ | Geant4 material G4_WATER; record the mean excitation energy Geant4 assigns |
| Outside phantom | Vacuum | Keeps the source definition the only input; air is an open decision (Section 11) |
| Source | Point source, 100 cm from phantom surface (SSD 100 cm) | Source model A |
| Emission | Photons emitted uniformly in solid angle within the pyramid that defines a 10 × 10 cm² field at the phantom surface | No collimator or capsule scatter in model A |
| Energies | 1.1732 MeV and 1.3325 MeV, equal probability | Co-60 gamma lines |
| Point of interest | Central axis, 5.0 cm depth | 5 g/cm² in water |
| Scoring volume | Cylinder on the beam axis, radius 1.0 cm, thickness 0.2 cm, centred at 5.0 cm depth | Sensitivity check S1 halves both dimensions |

Source model B, added in a later phase, replaces model A with a Co-60 unit spectrum or phase-space file that includes capsule and collimator scatter. Use only a source with documented provenance, and record the reference in `refs/refs.yaml`.

**Normalization.** Report dose per unit fluence. Define Φ₀ as the primary photon fluence, both lines combined, at the location of the point of interest (105 cm from the source) with the phantom removed. Compute Φ₀ analytically from the source definition. The headline dose quantity is D/Φ₀ in Gy·cm².

## 3. Observables: the five layers

Each layer is one step in the chain from photon to dose. For every observable, the Monte Carlo scores it in the R0 scoring volume and the AI is asked for the same quantity in the same units and bins.

**Binning for comparison.** Score Monte Carlo spectra in 10 keV bins and rebin to the comparison grid. The comparison grid uses 50 keV energy bins (photons 0 to 1.40 MeV; electrons from 10 keV to 1.20 MeV) and 10 equal bins in cos θ from −1 to 1, where θ is measured from the beam axis. Store the bin edges in one config file shared by the Monte Carlo and AI code.

### Layer 1. Photons arriving at the point

- Photon fluence spectrum Φ_γ(E), split into primary (no interaction since leaving the source) and scattered.
- Scattered fraction of photon fluence and of photon energy fluence.
- Fluence-weighted mean photon energy.
- Angular distribution of photon fluence, primary and scattered separately.

Method: track-length estimator (sum of track lengths in the volume divided by its volume). Tag each photon as primary or scattered with track user information.

Checks: primary fluence consists only of the two lines. Scattered fluence has no photons above 1.3325 MeV. Primary fluence for each line equals Φ₀ (that line's share) × exp(−μ · 5 cm) to within statistics, with μ from the cited NIST attenuation coefficients.

### Layer 2. Photon interactions at the point

- Fraction of photon interactions in the scoring volume by process: incoherent (Compton), photoelectric, coherent (Rayleigh), pair production.
- The same fractions weighted by energy transferred to charged particles.
- Both of the above, split by primary and scattered incident photon.

Method: tally process names for every photon interaction inside the volume, with the incident photon energy.

Checks: pair production only from photons above 1.022 MeV. Triplet production is impossible below 2.044 MeV.

### Layer 3. Electrons set in motion

- Initial kinetic energy spectrum of electrons and positrons created in the scoring volume by photon interactions, split by process and by primary or scattered parent photon.
- Mean initial kinetic energy.
- Distribution of initial direction (cos θ).

Delta rays, meaning electrons created by other electrons, are excluded here and belong to Layer 4.

Checks: Compton recoil electrons from primary photons end at the Compton edges, 0.963 MeV for the 1.1732 MeV line and 1.118 MeV for the 1.3325 MeV line (from Compton kinematics with m_e c² = 0.511 MeV; confirm in a unit test).

### Layer 4. Electrons slowing down

- Electron fluence spectrum Φ_e(E) in the scoring volume, all electrons including delta rays, from 10 keV upward. Positrons scored separately.
- Contribution of delta rays to Φ_e(E).
- Angular distribution of electron fluence.
- Ratio of radiative to collision energy loss for electrons in the volume.

Method: track-length estimator. Electron energy changes along a step, so distribute each step's length across the energy bins it spans, assuming linear energy loss along the step. Do not bin by pre-step energy alone.

### Layer 5. Energy deposited

- D/Φ₀: absorbed dose to water in the scoring volume per unit Φ₀ (Gy·cm²).
- Fraction of dose deposited by electrons whose parent photon was scattered.
- Kerma K and collision kerma K_col at the point, computed from Φ_γ(E) with tabulated μ_tr/ρ and μ_en/ρ (cite the source table).
- β = D/K_col.
- Radiative yield g = 1 − K_col/K.
- Spencer-Attix water-to-air stopping-power ratio s_w,air with Δ = 10 keV, including the track-end term, computed from Φ_e(E). Use one stopping-power data set for both media and record which one.
- Percent depth dose at 5 cm, PDD(5) = 100 × D(5 cm)/D(d_max). This is the recall item. Score it with a central-axis depth scan in 1 mm slabs from the surface to 10 cm.

**Expected values for sanity checks.** Humza must verify each against its cited source before it is used.

- s_w,air ≈ 1.133 for Co-60 (IAEA TRS-398, based on ICRU 37 stopping powers). ICRU 90 data shift this by a few tenths of a percent.
- β slightly above 1, on the order of 0.5% for Co-60 beyond d_max.
- Compton edges as in Layer 3.
- PDD(5) for source model A will differ somewhat from published Co-60 unit data (BJR Supplement 25), because model A has no head scatter. Document the difference rather than tuning to remove it.

## 4. Reference Monte Carlo

### 4.1 Geant4

- **Build.** Pin the latest stable Geant4 patch release at build time. Record the version and every data library version (G4EMLOW and others) in each run manifest. Build multithreaded.
- **Application.** Write a dedicated application for this study. Do not use TOPAS or GATE in phase 1, so that every scorer is visible in the code.
- **Physics.** Use G4EmStandardPhysics_option4 as the primary list. Run G4EmLivermorePhysics and G4EmPenelopePhysics as comparisons. Record the multiple-scattering models each list uses.
- **Production cuts.** Use thresholds equivalent to 10 keV or lower for electrons and photons in a region that extends at least 0.5 cm beyond the scoring volume on every side, or in the whole phantom if runtime allows.
- **Cut independence (check S2).** Repeat the Layer 4 and Layer 5 quantities with the threshold at 10 keV and at 5 keV. Quantities defined above Δ = 10 keV must not change beyond statistics.
- **Fano cavity test.** Run the Geant4 Fano cavity example (extended/medical/fanoCavity) with the chosen physics lists and step settings. Report the result before any production run.
- **Statistics.** Use at least 10 independent batches with distinct seeds, or the history-by-history method. Targets (1σ):
  - D/Φ₀ ≤ 0.1%;
  - β ≤ 0.2%;
  - spectral bins holding at least 1% of the total ≤ 1% each.
- **Event sample.** Record at least 10⁵ individual photon interactions in the scoring volume as the reference for the sampled-events task (Section 5.2): incident energy, process, scattered photon energy and angle, electron kinetic energy and angle.
- **Compute.** Measure throughput first and estimate CPU-hours for each run. If a run would take more than about 24 hours on the MacBook, propose cloud CPU with a cost estimate and wait for approval.

### 4.2 EGSnrc code-to-code benchmark (geometry B0)

The EGSnrc RZ user codes need cylindrical symmetry, so the cross-check uses a separate benchmark geometry B0:

- circular field with the same area as 10 × 10 cm² at the surface (radius 5.64 cm);
- cylindrical water phantom, radius 15 cm, depth 30 cm;
- same source energies, scoring cylinder, and vacuum outside the phantom.

Run Geant4 in B0 as well, and compare every quantity both codes can score:

- DOSRZnrc for dose;
- FLURZnrc for photon and electron fluence spectra;
- SPRRZnrc for s_w,air.

Report differences with combined uncertainties. Investigate any difference larger than 0.5% in D/Φ₀ or s_w,air before AI runs begin. Humza can change this threshold.

### 4.3 Comparison with published data

Compare R0 Monte Carlo results with:

- NIST attenuation and energy-absorption coefficients for water at the two line energies;
- the expected values in Section 3;
- published Co-60 percent depth dose for the matching SSD and field size.

Each published number goes into `refs/refs.yaml` with its source and table, flagged unverified until Humza checks it.

## 5. AI conditions

### 5.1 Models

Use at least three frontier models and two open-weight models. Record exact model identifiers, access route, access dates, and sampling and reasoning settings. The model list is an open decision (Section 11).

### 5.2 Conditions

| Condition | What the model gets | What it isolates |
|---|---|---|
| A0, recall | Problem statement only, no tools | Physics held in the model |
| A1, compute | A0 plus a Python sandbox (numpy, scipy; no network; no transport codes installed) | Whether the model can build its own calculation or simulation |
| A2, compute with data | A1 plus NIST XCOM and ESTAR tables as files | Method separated from knowledge of the data |

Run model-written code in an isolated container with no network access and fixed limits (proposed: 4 cores, 10 minutes per call).

**Sampled-events task**, run in A0 and A1. The model returns individual photon interaction events at the point with the same fields as the Monte Carlo event sample. Request 200 events per call and make 5 calls per model and condition.

### 5.3 Prompting

- One frozen problem statement containing the full R0 definition, the observables, units, the bin edges, and a JSON output schema. Prompts are versioned in `ai/prompts/`.
- Debug formatting only on a pilot configuration P0 (proposed: 2 MeV monoenergetic photons, 10 cm depth, same field). P0 results are excluded from analysis.
- No prompt changes after R0 AI runs begin.
- Run each model and condition k = 5 times.
- On malformed JSON, retry once and log the retry. Count refusals, parse failures, and failed sandbox calls, and report them.
- AI prompts never contain Monte Carlo results or reference values.

### 5.4 Cost

Estimate API cost for the full run matrix after P0 and get Humza's approval before M4.

## 6. Comparison and analysis

- **Scalars.** Relative difference (AI − MC)/MC with the Monte Carlo uncertainty. Summarize each model and condition by the median and range over the k repeats.
- **Binned spectra and angular distributions.** Normalize both distributions. Report the Wasserstein-1 distance on bin centres and the Jensen-Shannon divergence, and plot AI against MC overlays.
- **Sampled events.** Two-sample Kolmogorov-Smirnov tests on one-dimensional marginals and the energy distance on the joint distribution. Also run per-event physical validity checks:
  - energy conservation;
  - consistency between scattered photon energy and scattering angle under Compton kinematics;
  - recoil electron energy no higher than the Compton edge;
  - pair production only above threshold.

  Report the fraction of physically invalid events separately from the distributional distances.
- **Breakpoint analysis.** For each model and condition, identify the first layer where the AI result falls outside tolerance. Humza sets the tolerances before M4. Proposed defaults:
  - 2% for Layer 5 dose quantities;
  - 5% for interaction and dose fractions;
  - for distributions, three times the distance between independent Monte Carlo batches.
- **Recall against compute (H3).** Error in PDD(5) against error in D/Φ₀, per model.
- **Primary against scatter (H1, H2).** Errors in primary-only quantities against errors in scattered-photon and electron-transport quantities.
- **Resolution floor.** Show the spread among the Geant4 physics lists and EGSnrc on every figure. AI differences inside that spread are reported as unresolved.

## 7. Later phases (out of scope for the first build)

- V1, variations around R0:
  - depths of 1, 10, and 20 cm;
  - fields of 5 × 5 and 20 × 20 cm²;
  - SSD 80 cm;
  - a bone-like or lung-like slab above the point;
  - an invented compound medium, so no table exists to recall.
- V2, source model B.
- V3, measured leg. An ion chamber reading at reference conditions, either from an ADCL setup or as a 6 MV analogue on a clinical linac with a k_Q correction. Humza leads the measurements. Claude Code prepares only the analysis side.
- V4, training. The same scoring pipeline becomes the measurement action in an agent training environment, under a separate plan.

## 8. Repository and provenance

```
co60-refpoint/
  README.md
  STUDY_PLAN.md          this file
  DECISIONS.md           every design choice, with date and reason
  STATUS.md              progress and open questions at each gate
  refs/refs.yaml         every external number with source, table, verified_by_humza flag
  config/bins.yaml       shared bin edges
  mc/geant4/             application source, macros, physics configurations
  mc/egsnrc/             B0 input files
  mc/runs/               run manifests and outputs, never deleted
  ai/prompts/            frozen prompts and JSON schema
  ai/harness/            API client and sandbox runner
  ai/raw/                raw model responses, immutable
  analysis/              scripts, figures, tables
  tests/
```

Each run manifest records:

- git commit;
- Geant4 or EGSnrc version and data library versions;
- physics list and production cuts;
- seeds and number of histories;
- wall time and machine.

Required tests:

- Compton kinematics function against the analytic formula, including the two edges in Layer 3.
- Attenuation coefficient interpolation against the NIST table values.
- Track-length estimator on a thin slab against the analytic uncollided fluence.
- Unit conversions at every output boundary. Geant4 internal units must never reach an output file; outputs use MeV, cm, g, and Gy.
- s_w,air calculation reproducing SPRRZnrc on B0 within combined uncertainty.

## 9. Milestones and review gates

Each milestone ends with a `STATUS.md` entry covering what was done, what failed, and open questions. Wait for Humza's approval before starting the next milestone.

**M0. Environment.**
- Pinned Geant4 build and EGSnrc installation.
- Basic Geant4 test examples run.
- Fano cavity test result.

Gate: Humza reviews the environment report.

**M1. Monte Carlo at R0 with option4.**
- All five layers scored.
- Checks S1 (scoring volume size) and S2 (cut independence).
- Analytic primary-fluence check.
- Statistical targets met.

Gate: Humza reviews the results.

**M2. Monte Carlo validation.**
- Livermore and Penelope runs.
- EGSnrc B0 benchmark.
- Comparison with published values.

Gate: Humza verifies all reference values in `refs.yaml` and accepts the Monte Carlo as the reference.

**M3. AI harness.**
- Schema, prompts, and sandbox built.
- P0 pilot on one inexpensive model.
- Cost estimate for M4.

Gate: Humza approves the prompts, model list, tolerances, and budget.

**M4. AI runs at R0.** Full matrix of models × conditions × k repeats, plus the sampled-events task.

**M5. Analysis.**
- Figures and tables for every analysis in Section 6.
- A results summary in plain technical prose.

No manuscript text unless Humza asks for it.

## 10. Rules for Claude Code

1. Do not invent numbers. Every external value (coefficients, published dose data, expected values) enters `refs/refs.yaml` with its source and stays flagged unverified until Humza checks it.
2. Never place Monte Carlo results or reference values in AI prompts.
3. Freeze prompts and scoring definitions once M4 starts. Any later change needs a `DECISIONS.md` entry and a rerun of every affected comparison.
4. Keep every run. Archive superseded runs with a stated reason instead of deleting them.
5. Ask before any paid API call or cloud compute, with a cost estimate.
6. Report failures, null results, and anomalies in `STATUS.md` with the same detail as successes.
7. Git commits are authored by Humza only. Do not add co-author trailers.
8. Keep secrets (API keys) in a gitignored `.secrets` file.

## 11. Open decisions for Humza

1. SSD 100 cm (default) or 80 cm for R0.
2. Vacuum (default) or air between source and phantom.
3. Model list and access route.
4. Breakpoint tolerances (defaults in Section 6).
5. Stopping-power data basis for s_w,air: ICRU 37, ICRU 90, or both reported.
6. Source for model B (published Co-60 unit spectrum or phase-space file).
7. Target venue, which affects reporting format.
