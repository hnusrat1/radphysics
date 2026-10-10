# Status

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
