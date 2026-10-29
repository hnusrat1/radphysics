# Decisions

Each entry: date, decision, reason. Entries marked **[Humza]** need his confirmation at the next gate.

## D001. Study location in the repository (2026-10-10)

The study lives in `co60-refpoint/` inside the `radphysics` repository, with the layout of STUDY_PLAN Section 8 below that directory.
Reason: the repository is named `radphysics` and may hold other studies; the plan's tree is rooted at `co60-refpoint/`.

## D002. Geant4 release (2026-10-10)

Geant4 11.4.3, git tag `v11.4.3`, commit `2ee379e6843e55b9c6bbb835d2a2b03cc998fef0`, dated 2026-10-01.
Reason: latest stable patch release on 2026-10-10. The only newer tag is `v11.5.0.beta`, which is a beta and not eligible.
Source obtained from github.com/Geant4/geant4 (the GitHub mirror of the CERN GitLab repository) because gitlab.cern.ch is not reachable from the build environment. The install script refuses to build any other commit.

## D003. Geant4 build options (2026-10-10)

Release build, `GEANT4_BUILD_MULTITHREADED=ON`, C++17, TLS model `initial-exec` (default), system expat, Qt, OpenGL and GDML off, examples installed.
Reason: plan requires a multithreaded build; visualization and GDML are not needed and add dependencies.

## D004. Geant4 data libraries from conda-forge (2026-10-10)

In the cloud build environment the data libraries were installed from conda-forge `noarch` packages instead of the CERN data server, which the environment's egress policy blocks. Each package recipe downloads the official CERN tarball and pins its sha256; both the package sha256 and the tarball sha256 are in `env/versions.lock.yaml`. Dataset versions match `cmake/Modules/G4DatasetDefinitions.cmake` of 11.4.3 exactly.

Installed: G4EMLOW 8.8, G4ENSDFSTATE 3.0, G4PARTICLEXS 4.2, PhotonEvaporation 6.1.2, RadioactiveDecay 6.1.2, G4ABLA 3.3, G4INCL 1.3, G4PII 1.3, G4SAIDDATA 2.0.
Omitted: G4NDL, G4TENDL, G4NUDEXLIB, RealSurface, G4CHANNELING. They serve neutron-HP, optical, and channeling physics, which this study does not use.

On the MacBook, use `DATA_SOURCE=cern` (the default), which lets CMake download the CERN tarballs and verify their MD5. **[Humza]** Confirm the MacBook can reach geant4-data.web.cern.ch.

## D005. EGSnrc release (2026-10-10)

EGSnrc `v2026`, commit `bd49a47aae3938f49f4100b8e018091861408b19`, dated 2026-09-28, configuration `linux`, gfortran 13.3. Compiled: tutor1 (smoke test), DOSRZnrc, FLURZnrc, SPRRZnrc.
Reason: latest tagged release. The B0 benchmark needs only the three RZ codes.

## D006. Fano cavity test configuration (2026-10-10)

The Fano test runs the unmodified Geant4 example `extended/medical/fanoCavity` with its `standard_opt4` constructor (`PhysListEmStandard_option4`).

Reason. A Fano test cannot use the stock `G4EmStandardPhysics_option4`, `G4EmLivermorePhysics`, or `G4EmPenelopePhysics` constructors. Fano conditions require (a) a Compton model that regenerates the primary photon so interaction sites are uniform, and (b) an ionisation model with the density-effect correction removed so wall and cavity have the same mass stopping power. The example supplies both (`MyKleinNishinaCompton`, `MyMollerBhabhaModel`).

What the test covers. Reading the 11.4.3 sources, the three production lists share the same electron transport settings, and `PhysListEmStandard_option4` reproduces them:

| Setting | option4, Livermore, Penelope (11.4.3) | fanoCavity standard_opt4 |
|---|---|---|
| e-/e+ msc model below 100 MeV | Goudsmit-Saunderson | Goudsmit-Saunderson |
| Mott correction | on | on |
| msc step limit type | fUseSafetyPlus | fUseSafetyPlus |
| msc skin | 3 | 3 |
| msc range factor | 0.08 | 0.08 |
| e-/e+ step function | 0.2, 10 µm | 0.2, 10 µm |
| lowest electron energy | 100 eV | 100 eV |
| energy-loss fluctuation model | Urban | Urban |

What the test does not cover. The lists differ in the ionisation model below 100 keV (Penelope in option4, Livermore in G4EmLivermorePhysics, Penelope up to 1 GeV in G4EmPenelopePhysics), in the bremsstrahlung model (Penelope in G4EmPenelopePhysics), and in photon models. The Fano test as built cannot exercise these, because removing the density effect requires a modified model for each. These differences are cross-section differences, which the Monte Carlo spread (secondary question) is designed to show, rather than transport-algorithm artefacts, which the Fano test targets. **[Humza]** Accept this coverage or ask for Fano-modified Livermore and Penelope ionisation models.

Other settings, all example defaults: 1.25 MeV photons (mean of the two Co-60 lines), wall 5 mm of G4_WATER on each side, cavity 2 mm thick and 1 cm radius of G4_WATER at 1 mg/cm³, production cut 10 km (no delta rays, CSDA), Compton cross-section factor 1000, no bremsstrahlung. Delta-ray thresholds are tested by check S2 in M1, not here.

The cavity happens to have the same dimensions as the R0 scoring volume (radius 1.0 cm, thickness 0.2 cm).

## D007. Negative control for the Fano test (2026-10-10)

Run the same geometry with the example's `standard_opt0` constructor (Urban msc with default step limitation).
Reason: a test is informative only if it can fail. Poon et al., Phys. Med. Biol. 50 (2005) 681 (cited in the example README) reported Fano failures for Geant4 condensed-history transport with default settings of that era. If opt0 also passes here at the achieved precision, the test has not shown it can discriminate. The control is reported as a sensitivity check, not as a candidate configuration.

## D008. Compute host for M0 (2026-10-10)

M0 ran in a 4-core cloud container (Intel Xeon 2.8 GHz, 15 GB RAM, Ubuntu 24.04) provided with the Claude Code session, not on the MacBook. No separately billed cloud compute was used. The M0 workloads are small (Fano runs of order 1 CPU-hour).
Consequence: throughput figures in M0 are for this container. Production estimates in M1 need throughput measured on the MacBook. **[Humza]** Decide where M1 production runs.

## D009. Python environment (2026-10-10)

Python 3.13 virtual environment with versions pinned in `env/requirements.txt`. External constants used in code are read from `refs/refs.yaml` by key (`analysis/refs.py`), so no number is typed into analysis code.

## D010. Commit authorship (2026-10-10)

Commits are authored and committed as Humza Nusrat, without co-author trailers, per STUDY_PLAN Section 10.7.

## D011. Extension of the Fano run (2026-10-10)

The first standard_opt4 Fano run (10 × 340 000 events) gave 1.0016 ± 0.0011, a deviation at 1.4σ. Thirty more batches with new seeds were run (`M0-fano-opt4-ext`, about 1.6 h in the container) and pooled with the first run.
Reason: at ±0.11% the test could not tell a 0.1% transport artifact from zero, and 0.1% is the plan's statistical target for D/Φ₀. Pooling is valid because every batch has the same number of events and the same settings. Both runs are kept and reported separately as well as pooled.

## D012. Comparison bin grid (2026-10-10, approved by Humza at the M0 gate)

Electron comparison grid: edges 0.010, 0.050, 0.100, ..., 1.200 MeV, so the first bin is 40 keV wide, plus one overflow bin 1.20 to 1.40 MeV. The overflow bin holds photoelectrons from the 1.3325 MeV line (about 1.33 MeV), which would otherwise be dropped silently. Photon and cos θ grids as in the plan. All edges are listed explicitly in `config/bins.yaml`. The Monte Carlo scores photons in 10 keV bins, electrons in 1 keV bins and cos θ in 40 bins. Every comparison edge is an edge of these grids, so rebinning is exact. The 1 keV electron grid also serves the Spencer-Attix integral at Δ = 10 keV.

## D013. Compute host for M1 (2026-10-10, delegated by Humza: "you figure it out")

M1 runs in the 4-core cloud container used for M0, with no separately billed compute. Each production run fits in a few hours there once variance reduction is applied (D016). The MacBook remains the reference machine for any future rerun; the install scripts and run specs reproduce every run.

## D014. Compton validity checks with bound electrons (2026-10-10, approved by Humza at the M0 gate)

All three production lists sample Compton scattering from bound electrons with Doppler broadening. The free-electron rule "recoil electron energy ≤ Compton edge" therefore fails for a small fraction of correct Monte Carlo events. The study reports two checks:

1. **Strict free-electron check**, as in the plan. Its failure rate is reported for the Monte Carlo reference and for every AI condition, so an AI failure rate is read against the Monte Carlo rate, not against zero.
2. **Bound-electron check**, the pass/fail rule for physical validity. A Compton event is valid if:
   - (a) energy is conserved within the binding energy of the most tightly bound shell of the target (oxygen K shell; value from the Geant4 atomic data, recorded in refs.yaml);
   - (b) the recoil electron energy does not exceed the free-electron edge by more than a tolerance δ_C;
   - (c) the scattered photon energy matches the Compton formula for its angle within δ_C.

   δ_C is the 99.9th percentile of the Compton departure from free-electron kinematics in the Monte Carlo event sample, rounded up to the next 10 keV. It is frozen before any AI event is scored. The departure is the larger of |E_out − E_KN(θ)| and the recoil energy above the edge. (Amended 2026-10-10: the maximum departure was first proposed; at 134 keV it would make the check too weak to matter. The Monash Compton model's Doppler tail reaches that far.) The Monte Carlo's own failure rate under the bound rules is therefore about 0.1% by construction, and AI failure rates are read against it.

The same per-event checks apply to Monte Carlo and AI events.

## D015. Water definition for the B0 benchmark (2026-10-10, delegated by Humza)

EGSnrc will use PEGS4 data generated for water matched to G4_WATER rather than the standard `521icru.pegs4dat`:
- same elemental composition by mass (from Geant4 11.4.3 G4NistMaterialBuilder);
- ρ = 1.000 g/cm³;
- I = 78.0 eV;
- AE = 0.521 MeV, AP = 0.010 MeV (10 keV kinetic thresholds);
- density-effect correction computed by PEGS4 for these I and ρ.

Reason: with standard ICRU 37 water (I = 75 eV) the stopping-power difference alone could move s_w,air by a few tenths of a percent, near the 0.5% investigation threshold. That would make the benchmark test the data rather than the transport. Air is treated the same way (G4_AIR composition and I = 85.7 eV). Stopping powers from the matched PEGS4 data are compared with Geant4's for the same materials before any B0 run, and the residual difference is reported. B0 runs themselves stay in M2.

## D016. Variance reduction in co60app (2026-10-10)

Analog transport cannot reach the plan's 0.1% target on D/Φ₀ in reasonable time. At R0 the per-history relative standard deviation of the dose is 48.7, so 0.1% needs 2.4 × 10⁹ histories, about 40 hours on 4 cores. Two unbiased techniques are used:

1. **Electron Russian roulette.** An electron born outside a region of interest (ROI) around the scoring volume survives with probability p = 0.05 and then carries weight 1/p.
   - The ROI extends 1.2 × the CSDA range of a 1.3325 MeV electron in water beyond every scored region. The application computes that range from the production physics (6.18 mm) and refuses to run if the ROI is too small; the factor 1.2 covers range straggling.
   - Survivors are transported normally, so their bremsstrahlung, which can reach the point, is kept with the right weight.
   - Positrons, photons, and electrons that descend from interactions in the scoring volume (needed for collision kerma) are never rouletted.
2. **Photon importance sampling.** Geant4's G4ImportanceBiasing runs in a parallel world of nested cylinders around the scoring volume, with importances 2, 8 and 32 relative to the rest of the world. Photons are split on entering a more important cell and rouletted on leaving it. The parallel world has no material and does not change the mass geometry or the physics.

Every tally is weighted. Measured efficiency, 1/(σ²T) for the dose:

| Setup | Efficiency |
|---|---|
| Analog | 4.8 |
| Roulette only | 10.5 |
| Roulette and importance sampling | 119 |

The layout was chosen from five tested layouts by this figure of merit. The implementation is checked by an analog run at matched settings (`M1-vrcheck-analog`) and by the thin-slab and analytic primary-fluence tests. The event sample records each interaction's weight; with importance sampling the weights inside the innermost cell are 1/32, apart from rare photons that come from rouletted electrons.

## D017. Development runs (2026-10-10)

Short runs made while debugging co60app ran in the session scratch directory and are not archived; they used code that was never committed. Every run that informs a result is made from committed code through `mc/tools/run_co60.py` and archived under `mc/runs/`. Runs made before their code was committed were moved to `mc/runs/_archived/` with a reason and rerun.

## D018. Stopping-power bases for s_w,air (2026-10-10)

Open decision 5 (ICRU 37, ICRU 90, or both) is unresolved, so s_w,air is reported on four bases, all with Δ = 10 keV and the Nahum track-end term, using the same scored electron fluence:

| Basis | Water | Air |
|---|---|---|
| icru37 | Bethe–Møller restricted collision stopping power, I = 75 eV | I = 85.7 eV |
| icru90 | as icru37, I = 78 eV | as icru37 |
| geant4 | G4EmCalculator restricted dE/dx from the production physics | same |
| pegs4 | EGSnrc PEGS4 EDEDX for media matched to Geant4 (D015): what EGSnrc will use in B0 | same |

- For icru37 and icru90, the density-effect corrections come from the ESTAR-computed tables shipped with EGSnrc.
- The implementation (`analysis/stopping.py`) reproduces Geant4 and PEGS4 to 0.05% for air above 100 keV, where all three use the same physics (test in `tests/test_mc_outputs.py`).

Reason: one data set for both media per basis, as the plan requires, while keeping open decision 5 open.

## D019. General gamma process switched off in option4 (2026-10-10)

G4EmStandardPhysics_option4 in 11.4.3 enables G4GammaGeneralProcess. In its energy zone from 2 m_e c² to 100 MeV that process selects only pair production, Compton, photoelectric and photonuclear; Rayleigh scattering is not sampled there (`G4GammaGeneralProcess.cc`, `PostStepDoIt`, case 2).

Both Co-60 lines lie in that zone, so primary photons could never scatter coherently. The first production attempt showed no Rayleigh events for primaries, where about 30 per batch were expected from the Rayleigh cross section that the same physics computes.

All co60app production runs therefore set `/process/em/UseGeneralProcess false` before initialization:
- photon interactions are then sampled by the separate processes with the same models;
- the only change in physics is that coherent scattering is restored above 1.022 MeV;
- G4EmLivermorePhysics and G4EmPenelopePhysics do not use the general process, so this also makes the three lists consistent for M2.

## D020. Electron track-length estimator: linear (plan) and CSDA-weighted (2026-10-10)

Layer 4 asks for each step's track length to be spread over the step's energy interval assuming linear energy loss along the step.

The problem: Geant4 transports an electron to rest in one step once its range falls below the step-function final range (10 µm in option4, about 20 keV in water). Over such a step the path per unit energy is 1/L(E), not constant. The linear rule therefore puts too much track length at low energy.

Evidence: with the linear rule, the dose rebuilt from the scored fluence, ∫Φ_E L_Δ dE, exceeds the scored dose by 6.4%. With track length spread as 1/L(E), using the restricted CSDA range R(E) = ∫dE/L(E) from the production dE/dx at the production threshold, the same closure gives 1.0007.

co60app scores both estimators:
- the plan's linear estimator, as `electron_fluence` etc.;
- the CSDA-weighted estimator, as `electron_fluence_csda` etc.

Analysis uses the CSDA-weighted spectrum for Layer 4 and for s_w,air and reports the linear one alongside. On s_w,air the two differ by less than 0.03%. **Approved by Humza at the M1 gate (2026-10-10).**

## D021. Mass-geometry importance sampling, built as an alternative (2026-10-10)

The main run's primary fluence came out 0.15% below the analytic value (STATUS, M1 anomalies). While parallel-world importance sampling was a suspect, I added a second implementation: the importance cells become nested G4_WATER cylinders in the mass geometry, clipped to the phantom, as in Geant4 example B01. There the importance process acts only where a step really ends on a geometric boundary.

- Selected per run spec with `importance_geometry: mass`; the launch flag is `--importance-mass`. The default stays the parallel world.
- The scoring cylinder must lie inside the innermost cell. Electrons that can reach the scoring volume never reach a cell wall (ROI radius 1.8 cm + CSDA range 0.62 cm < 2.5 cm, and likewise in z), so the added boundaries cannot change the scored electron transport.
- Built into a separate binary (`co60app.dev-8440e7c`); the frozen production binary is unchanged.

Not adopted. The photon-only checks found no bias from the parallel implementation (STATUS), so production keeps D016. The option and its check spec (`M1-check-primary-ismass-prodgeom`) stay in the repository.

Outcome (2026-10-11): `M1-check-primary-ismass-prodgeom` (8 batches, 4×10⁷ histories) gives primary/analytic 0.9987 ± 0.0012, against 1.0005 ± 0.0011 for the parallel world in the same geometry (difference −1.1σ). Both agree with the analytic value. The mass-geometry option stays available but is not used.

## D022. Replicate of the main run, with the pooling rule fixed in advance (2026-10-10)

The main run's primary-fluence deficit (0.9985 ± 0.0003 against the model μ, 4σ) was not reproduced by any diagnostic. `M1-R0-option4-rep2` repeats the main run exactly (same spec, same production binary) with new seeds (seed_base 6111).

Rule, written before the replicate finishes:
- If the replicate agrees with the main run within 3 combined standard errors on D/Φ₀ and on the primary fluence ratio, the two are pooled (20 batches) for every R0 result, and both are also reported separately.
- If they disagree on either, they are not pooled; both are reported, and the difference is treated as an unresolved systematic until explained.

Outcome (2026-10-10 21:44 UTC): the replicate agrees with the main run.
- D/Φ₀: z = +0.04.
- Primary fluence, tracking μ: 1.1732 MeV z = +1.36, 1.3325 MeV z = −0.03.
- The replicate's own primary fluence is 0.9996 ± 0.0006.

The runs are pooled (20 batches, 2.4 × 10⁸ histories); the pooled primary fluence is 0.9992 ± 0.0005 (−1.7σ). The main run's deficit was a chance fluctuation, made to look significant by its unusually small batch spread (D023). The replicate's batch spread matches its history-by-history variance (ratio 1.06).


## D023. Uncertainties: the larger of the batch spread and a second estimate (2026-10-10)

The plan's uncertainty is the standard error over at least 10 independent batches. With 10 batches that estimate is itself uncertain by about ±24% (1σ), and it can come out far too small by chance.

The main run shows this. Its batches agree with each other better than their own history-by-history variance allows: the batch spread of the total photon track length is 0.64 of the history-based value, D/Φ₀ 0.71, the primary fluence 0.43 of its counting-statistics value. S2, the analog run and the photon-only checks show ratios near 1. The event samples share no records between batches, so the batches are not reusing random streams. The main run's small spread is a chance outcome (p ≈ 0.02–0.07 per quantity).

From M1 on, analysis/m1.py reports, beside the batch standard error:
- `sem_hbh` for quantities with per-history second moments (D, K, total photon fluence): the standard error from the variance of the per-history scores;
- `sem_counting` for the primary fluence: the Poisson error of the number of primary crossings (track length over thickness), a floor for the track-length estimator;
- `sem_conservative`, the larger of the batch standard error and these.

Statistical targets, run-to-run comparisons and the primary-fluence check use `sem_conservative`. Quantities without a second estimate (β, spectra, s_w,air) keep the batch standard error.

Effect: the main run's D/Φ₀ relative uncertainty is 0.093% (target 0.1%, met). Its primary-fluence deficit against the tracking μ is 1.9σ instead of 4.4σ. **Approved by Humza at the M1 gate (2026-10-10).**

## D024. s_w,air reference for scoring AI answers: ICRU 90 primary, ICRU 37 alongside (2026-10-10, M1 gate)

Decided by Humza at the M1 gate, on my recommendation, before any AI answer exists.

AI answers for s_w,air at R0 are scored against the Spencer–Attix value (Δ = 10 keV) from the Monte Carlo electron spectrum:
- primary: ICRU 90 stopping powers (I = 78 eV);
- reported alongside: ICRU 37 stopping powers (I = 75 eV).

Reasons:
- ICRU 90 is the current reference data, and its I-value matches the water in the Geant4 transport, so spectrum and stopping powers are consistent.
- Many models will quote the TRS-398 value, which rests on ICRU 37. Reporting the distance to both separates "quoted the protocol" from "got the physics wrong".
- The Geant4-basis value is not used for scoring: it reflects option4's Penelope ionisation model below 100 keV, not reference data.
- To check: whether the 2024 revision of TRS-398 adopted ICRU 90. If it did, ICRU 90 is also the protocol-consistent choice.

This is part of the scoring definition and freezes with it at M4 (plan rule 3).

## D025. Scoring-volume boundary: systematic on D now, boundary-free dose scoring from M2 (2026-10-10, M1 gate)

Decided by Humza at the M1 gate, on my recommendation.

- **M1.** D at R0 is quoted from the scoring cylinder, with the boundary effect bounded by the boundary-free runs (`M1-check-boundary-free-R0` plus the 20-batch extension `-ext`, pooled) as a systematic. If the pooled ratio stays consistent with 1, the systematic is its uncertainty; if it does not, D at R0 is corrected to the boundary-free value and the correction is reported.
- **M2 onward.** Dose at R0 is scored by position with no geometry boundary within an electron range of R0. The scoring cylinder stays for quantities that need volume-limited steps (fluence spectra, interaction records), and those are checked once against a boundary-free variant.

Reasons:
- The dose next to a mass-geometry boundary is distorted by up to ±3% in water–water geometry (STATUS M1), from the electron step limit near boundaries.
- EGSnrc crosses boundaries with its own algorithm. A boundary at the scoring volume would mix boundary handling into the code-to-code comparison of M2.

Before M2, `RunAction::CheckRoi` also requires the 1.2 × CSDA buffer in front of a depth scan (it previously required only that the ROI start at or before the scan).

## D026. Fano result, M2 batch count, refs.yaml verification order (2026-10-10, M1 gate)

Decided by Humza at the M1 gate, on my recommendation.

- **Fano +0.16 ± 0.06%** is accepted as a bound on interface artifacts. The step-size diagnostics (range factor 0.04, step function 0.05 / 1 µm) are reviewed before M2 production. If the deviation shrinks with tighter stepping, M2 production uses the tighter setting, and the boundary artifact is re-measured with it.
- **M2 production** uses at least 20 batches per run, so that the batch standard error is itself better determined (D023).
- **refs.yaml** is verified by Humza in this order:
  1. Co-60 line energies;
  2. TRS-398 s_w,air (and whether its 2024 revision uses ICRU 90);
  3. I-values;
  4. constants used in normalisation;
  5. the remaining entries, `recall` entries first.

  A published PDD(5) for Co-60, 10 × 10 cm², SSD 100 cm is to be added as an external check of the depth dose.

Outcome for the first step-size diagnostic (2026-10-11): with the msc range factor at 0.04 (production 0.08), the Fano ratio is 1.00001 ± 0.00088 (20 batches, `M1-fano-diag-rfact004`), against 1.00157 ± 0.00058 with production stepping. The deviation disappears with tighter stepping, although the difference between the two is only 1.5σ. Unless the step-function diagnostic (`M1-fano-diag-stepfn`, running) points the other way, M2 production uses range factor 0.04, after a throughput measurement, and the boundary artifact is re-measured with it.

Outcome for the second step-size diagnostic (2026-10-11): with the step function at 0.05, 1 µm (production 0.2, 10 µm), the Fano ratio is 0.99921 ± 0.00073 (20 batches, `M1-fano-diag-stepfn`). Both tightenings remove the production deviation; neither result differs from 1 by more than 1.1σ. **M2 production uses msc range factor 0.04 for every Geant4 run (D028 item 8).** The step function stays at the option4 default, since the range factor alone already brings the ratio to 1.0000 and is the cheaper change.

## D027. R0 dose reference after the boundary-free extension (2026-10-11)

D025 said the cylinder's D at R0 is replaced by the boundary-free value if the two are not consistent, but set no threshold. Applied: replace if they differ by 2 combined standard errors or more (95%).

Outcome:
- Cylinder over boundary-free D at R0 = 0.99759 ± 0.00109 (−2.2σ). Boundary-free: `M1-check-boundary-free-R0` plus `-ext` (30 batches, the main run's variance reduction), combined by inverse variance with the analog depth scan `M1-check-pdd-cyl8` (4 batches, about 1% of the weight). Cylinder: main run plus replicate, 20 batches.
- **The R0 dose reference becomes the boundary-free value: D/Φ₀ = 5.3713 × 10⁻¹² Gy cm² ± 0.087%.**
- β = D_boundary-free / K_col = 1.0069 ± 0.0011.
- The cylinder values (5.3584 × 10⁻¹² Gy cm², β = 1.0045) are reported alongside.
- Kerma, K_col, interaction shares and fluence spectra come from photon interactions or track lengths and are unchanged. s_w,air is a ratio of spectral averages: it changed by less than 0.001% between scoring volumes of different size (S1), so any boundary effect on it is negligible.

Caveats:
- −2.2σ is close to the threshold. The boundary-free uncertainty is from batch spread only, because the depth tally has no per-history moments; with 30 batches that estimate is good to about ±13%.
- M2 re-measures R0 with boundary-free scoring (D025) and, if adopted, tighter stepping (D026), so the M1 dose values are provisional.

**[Humza]** Confirm the 2σ threshold and the switch to the boundary-free value.

## D028. Open decisions delegated to Claude (2026-10-11)

Humza, 2026-10-11: "just pick your best recommended choices, continue and dont stop until you have the full study ready for me to review!" The choices below are my recommendations, taken under that delegation. Each can be reversed at review; anything that depends on one is listed with it. Two things are not delegable under Section 10 and stay with Humza: verifying `refs.yaml` (rule 1) and approving paid API spend (rule 5).

1. **D027 threshold.** 2σ confirmed. The R0 dose reference is a boundary-free dose (D025). M2 re-measures it with the M2 stepping (item 8), and that value replaces the M1 one.
2. **R0 stays as planned (Section 11 items 1, 2).** SSD 100 cm, vacuum between source and phantom. Air would add only a 0.1%-level attenuation and scatter term to the normalization. It would also give the AI one more thing to get wrong that is not part of the physics chain.
3. **s_w,air basis (Section 11 item 5).** As D024: ICRU 90 primary, ICRU 37 alongside.
4. **Breakpoint tolerances (Section 11 item 4).** The plan defaults, made precise:
   - Layer 5 dose quantities (D/Φ₀, K/Φ₀, K_col/Φ₀, β, s_w,air, PDD(5)): |AI − MC|/MC ≤ 2%.
   - Scalars in Layers 1–4 (mean energies, scattered fractions) and all interaction and dose fractions: |AI − MC| ≤ max(5% of MC, 0.005 absolute). The absolute floor keeps a 5% relative test from being impossible for fractions near zero (photoelectric, pair, g).
   - Distributions: W1 and JSD (Section 6) ≤ 3 × the same distance between two independent halves of the Monte Carlo batches.
   - Resolution floor (Section 6): a difference is "unresolved" when it lies inside the Monte Carlo spread, defined as the largest |list − option4| among option4, Livermore and Penelope at R0, plus twice its standard error. EGSnrc enters through the B0 benchmark.
   - A layer "breaks" when at least one of its quantities fails and is resolved. The breakpoint is the first such layer.
5. **Models (Section 11 item 3).**
   - Three frontier models, one each from Anthropic, OpenAI and Google: each provider's most capable generally available model on the M4 access date, through the provider's own API.
   - Two open-weight models from different labs, chosen among the leading open-weight releases on that date, with pinned weights. Current candidates are DeepSeek and Moonshot (Kimi), with Qwen as the alternate.
   - Exact model identifiers, access route, dates and settings are recorded per call in `ai/raw/` and frozen in `ai/models.yaml` when M4 starts. Humza confirms the list at launch.
   - Sampling: the provider's default temperature. The reasoning or thinking setting is the provider's highest generally available level, recorded per call.
6. **Source model B (Section 11 item 6).** Deferred to V2 as the plan says. Recommended source: the Mora, Maio and Rogers (1999) Co-60 unit spectrum, distributed with EGSnrc as `HEN_HOUSE/spectra/egsnrc/mora_10x10_60co_ssd80.ensrc`, entered in `refs.yaml` unverified.
7. **Venue (Section 11 item 7).** Medical Physics as the primary target. The Monte Carlo part follows the AAPM TG-268 reporting checklist and the AI part reports the TRIPOD-LLM items that apply. This changes no method; it fixes what the M5 tables report.
8. **M2 stepping (D026 outcome).** As D026 decided: the step-size diagnostics remove the Fano deviation, so every M2 Geant4 run uses msc range factor 0.04 (option4, Livermore, Penelope, B0). The new R0 reference is M2-R0-option4 (layers) plus M2-R0-option4-bf (dose). The M1 runs (range factor 0.08) stay and give the step-size sensitivity of every observable. The boundary artifact is not re-measured with a cylinder at range factor 0.04: from M2 the dose at R0 is scored without a nearby boundary, so the artifact no longer enters any reference value.
9. **Published PDD(5).** No source for Co-60, 10 × 10 cm², SSD 100 cm is reachable from this environment. BJR Supplement 25 is not online, and physics.nist.gov and the publishers' PDFs are blocked. The entry stays null for Humza to fill. It is a context check only (Section 3 expects model A to differ from unit data); H3 uses the Monte Carlo PDD(5).
10. **XCOM and ESTAR files for condition A2.** physics.nist.gov is unreachable. Two stand-ins are built in `ai/data/`:
    - an XCOM water table from the XCOM elemental data distributed with EGSnrc, using XCOM's mixture rule;
    - an ESTAR-method water table, from ESTAR's density-effect corrections and the Seltzer–Berger bremsstrahlung data distributed with EGSnrc.

    Both are labelled as reproductions, not NIST downloads. They are replaced by the genuine NIST tables before M4 if Humza can download them; `ai/data/validate_tables.py` checks either version.

## D029. One access route for every model: OpenRouter (2026-10-12)

Humza, 2026-10-12: "im just giving you one openrouter key". Every model slot goes through OpenRouter's OpenAI-compatible
Chat Completions API with one key, `OPENROUTER_API_KEY`, kept in the gitignored `.secrets` file or as an environment
secret. Only `openrouter.ai` needs to be reachable.

- Reasoning is requested with OpenRouter's unified `reasoning: {effort: high}` setting. Tool calls use the
  OpenAI function-calling format, which OpenRouter translates for each upstream provider.
- Each raw response records the model OpenRouter served and its upstream provider. A call answered by a different
  upstream than expected is visible in `ai/raw/`.
- Cost: OpenRouter passes through provider list prices plus its own fee. The P0 pilot measures the actual charge
  before Humza approves the M4 budget.
- Trade-off: one route is simpler and gives one bill. The cost is that OpenRouter, not the provider, decides some
  request handling, such as fallbacks and caching. The pilot checks that each frontier slot is served by its own
  provider.

## D030. M4 model list: the fast, low-cost tier of each provider (2026-10-12)

Humza, 2026-10-12, named the five OpenRouter models for M4 and set a budget rule: run the P0 pilot, report the
measured cost, and run the full R0 matrix only if it fits under $10.

| Slot | OpenRouter model | Pinned release (OpenRouter canonical slug) | Listed price, $/Mtok in / out |
|---|---|---|---|
| frontier_anthropic | `anthropic/claude-haiku-5.5` | `anthropic/claude-haiku-5.5-20261007` | 0.10 / 0.50 |
| frontier_openai | `openai/gpt-6-luna` | `openai/gpt-6-luna-20260922` | 0.10 / 0.50 |
| frontier_google | `google/gemini-2.5-flash-lite` | `google/gemini-2.5-flash-lite` | 0.10 / 0.40 |
| open_weight_1 | `deepseek/deepseek-v4-flash` | `deepseek/deepseek-v4-flash-20260423` | 0.0173 / 1.28 |
| open_weight_2 | `qwen/qwen3.7-flash` | `qwen/qwen3.7-flash-20260727` | 0.03 / 0.13 |

Prices are from OpenRouter's public models endpoint on 2026-10-12 (`ai/prices.yaml`).

This departs from D028 item 5 in two ways:
- **Not the most capable models.** Each is its provider's small, fast tier: Haiku below Sonnet and Opus, GPT-6 Luna
  below Sol, Gemini 2.5 Flash-Lite below the 3.x Flash and Pro models. The study then measures where this tier
  breaks, not where the frontier breaks. Results must be reported under that name.
- **open_weight_2 is not open-weight.** OpenRouter lists DeepSeek V4 Flash with published weights
  (`deepseek-ai/DeepSeek-V4-Flash`) but gives no weights for Qwen3.7 Flash. The slot keeps its name because the
  scorer keys on it; the model is reported as API-only.

Reasoning stays at `effort: high` for every slot (D028 item 5). Each request also sets `usage: {include: true}`, so
every response carries OpenRouter's charge. `ai/harness/cost.py` sums the charges and projects them to R0.

## D031. Fixes from the P0 pilot (2026-10-12)

The pilot (`ai/raw/P0-pilot/`, excluded from analysis) found the following. The prompt files are unchanged, and
their hashes still match `MANIFEST.yaml`.

| Found in P0 | Fix for R0 |
|---|---|
| Asked for exactly 200 events, models returned 140 to 243 (14 answers) | The harness accepts 150 to 250 events (`EVENT_COUNT_ACCEPTED`). The prompt still asks for 200. The scorer pools events, so the count does not enter any statistic except the sample size |
| Haiku spent its whole default output (65,536 tokens) on reasoning and wrote no answer | `max_tokens: 128000` for Haiku, the model's maximum |
| Qwen answered a retry with the JSON schema itself | The retry text asks for "your answer, not the schema itself" |
| Haiku was served by "Claude Platform on AWS", not Anthropic (D029 check) | `provider: {order: [anthropic]}`, with fallbacks allowed and recorded per call |
| DeepSeek was served by a mix of fp4 and fp8 hosts | `provider: {quantizations: [fp8]}`, the release's native precision |
| GPT-6 Luna returned `"events": []` with a statement that it cannot sample without a transport code (all 10 events calls in A0 and A1) | None. This is a result. `analysis/m5.py` counts a parse failure whose last reply is valid JSON with no content as a refusal (STUDY_PLAN 5.2) |
| Gemini and Qwen sometimes ran past 200 events until the output limit (80,000 to 116,000 characters) | None. Both are already at their maximum output; such answers stay parse failures |

A1 and A2 tool use worked for every model. Every frontier slot is now served by its own provider.

## D032. M4 launch: caching for Haiku, budget, freeze (2026-10-12)

**Pilot cost.** The measured P0 charge was $3.90 for 64 conversations, plus $0.38 for one cached Haiku re-run and
the charge for one DeepSeek conversation cut off by a job time limit (`ai/raw/P0-pilot-killed/`). Six DeepSeek
pilot conversations were still running at launch.

**Projection.** Projected from the pilot, R0 cost $14.33, above Humza's $10 limit. Haiku was $12.77 of that: in
the tool conditions every turn re-sends the conversation, and Anthropic bills a request of 100,000 prompt tokens
or more at 5× the base price for input and output alike.

**Fix: prompt caching for the Haiku slot (`prompt_cache: true`).** Breakpoints sit on the system prompt and the
newest message. What the model sees is unchanged. Measured on Haiku's P0 A2 observables conversation: $0.38 with
caching against $1.87 without, with 95% of 3.5 million prompt tokens read from the cache. With that ratio
applied to Haiku's tool conditions, R0 projects to $4.72. Groups with no DeepSeek pilot data take the highest mean
of the other non-Haiku models.

**Launch.** R0 runs under Humza's rule ("run the full R0 matrix if it fits under $10"), with a hard stop:
`--budget 9.5` stops new conversations once R0's measured charge reaches $9.50. The headroom covers the five
conversations in flight. `ai/models.yaml` and `ai/prompts/v1/MANIFEST.yaml` are frozen as of this entry.

## D033. OpenRouter errors inside HTTP 200 replies (2026-10-12)

OpenRouter reports some upstream failures inside an HTTP 200 body: `finish_reason: "error"` with an error object,
for example 429 "temporarily rate-limited upstream" or 504 "Upstream idle timeout exceeded". The harness read these
as empty replies, so they became parse failures after the one retry.

- **Fix.** `providers.in_band_error` detects them. Retryable codes (408, 409, 429, 5xx) are retried with backoff
  (10 s doubling, 6 attempts). Anything else, or a persistent error, raises and the conversation is recorded as
  `api_error`. Checked on every stored response: it flags exactly the 17 error replies and none of the other 368.
- **R0.** Stopped about 3 minutes after launch. Seven conversation directories with no model output were moved to
  `ai/raw/M4-R0-void/`: two Gemini conversations whose only replies were in-band 429s, and five cut off by the stop
  before any reply. None was charged. R0 restarted with the fix.
- **P0 reading.** 13 Gemini pilot replies were in-band 429 or 504 errors, so some of Gemini's pilot "parse
  failures" were rate limits, not model behaviour. The pilot is excluded from analysis and is not rerun.

## D034. Parser fallback on malformed fenced JSON: kept as launched for R0 (2026-10-12)

When an answer's fenced JSON block does not parse, `parse.extract_json` falls back to the last balanced `{...}` in
the text. Inside a broken answer, that is a nested object. The validator then reports "method missing" instead of
the JSON error, and the one retry tells the model the wrong thing. Seen in R0 for DeepSeek (A0 observables r2: a
number written `.0008`; r4: U+FFFC characters between keys) before either reached the retry.

- **R0 keeps the parser it launched with.** The running processes have it loaded. Changing it would give some
  repeats a different retry message from others. The outcome of such an answer is a parse failure either way.
- **Fixed after R0.** With a fenced block present the parser reports the decode error and does not fall
  back. Re-scoring R0 with the fix gives identical results.
- **Scorer.** `analysis/m5.py` counts a refusal only for a top-level answer (one that has `method`), so a nested
  fallback object is no longer read as an empty answer.

## D035. R0 execution: parallel processes, API-error reruns (2026-10-12)

- **Parallel processes.** R0 started as one process per model. Once DeepSeek and Haiku had reached their tool
  conditions, extra processes were started for their remaining blocks: A1 events and A2 observables, plus A2
  observables for Qwen. Each process skips a conversation whose directory exists, so every conversation runs once.
  The order of conversations changes; nothing a model sees does.
- **API errors are rerun.** Gemini had four `api_error` conversations:
  - two in-band errors with no error code;
  - one 429 that outlasted 6 retries;
  - one upstream 504 idle timeout.

  An API error is an infrastructure failure, not model behaviour, so these four were moved to
  `ai/raw/M4-R0-void/<slot>/<name>.api_error` and rerun. Together with the empty attempts in D033, they carry
  $0.0045 of charge. For the reruns the harness retries in-band errors without a code as well, with backoff of 20
  s doubling over 8 attempts. This applies to the reruns and to conversations started after 18:20 UTC; earlier
  ones used D033's 6 attempts from 10 s. Only the API retry differs; the prompts, the parser and the model's
  retry text do not.

## D036. Code audit of the A1 and A2 answers (2026-10-12; exploratory, not in the analysis plan)

Humza asked whether models that write their own Monte Carlo defeat the purpose of A1. To see what each answer
rests on, all 50 A1/A2 observables conversations were classified, and the 40 with code were audited against a
physics checklist (`analysis/m4_audit_spec.md`).

- **Who did it.** Four Claude subagents read the transcripts, one per model. Every claim carries an exact quote
  from the code or its output, and each subagent checked its quotes against the files programmatically. I
  checked eight claims against the raw files myself; all were confirmed:
  - GPT-6 Luna's typed scatter constant;
  - its C++ compile;
  - its cylinder-only interaction bug;
  - Qwen's 1.602e-13 conversion;
  - Qwen's Monte Carlo claim with no random sampling in the code;
  - DeepSeek's "validated" Monte Carlo claim;
  - DeepSeek's Compton envelope;
  - Haiku's backscatter exit.
- **Output.** The audit is in `analysis/results/M4_method_audit.json`; scores by group come from
  `analysis/m4_methods.py` and are in `analysis/results/M4_methods.json`.
- **Photon-field category.** My judgement from the audit, one per conversation:
  - `simulated`: the scattered photons come from a photon-transport simulation the model ran;
  - `simulated_with_major_bug`: three runs, each with its bug named in the file;
  - `assumed`: the scatter fluence and spectrum are typed or formula values;
  - `no_code`: Gemini, which made no tool call.
- **Sandbox finding.** The plan defines A1 and A2 as a Python sandbox, but the sandbox exposed the container's
  compilers (`g++`, `clang`). GPT-6 Luna compiled and ran C++ in three R0 conversations (A1 events r3, A2
  observables r2 and r5) and in one pilot conversation. Isolation held: no network, no transport code or results,
  and the same time and core limits. The answers are kept and the compiler use is reported. For future runs,
  either remove the compilers from the sandbox or state that they are available.
- **Status.** This analysis was added after seeing the data. It is reported as exploratory and does not change
  any verdict in `M5_M4-R0.json`.

## D037. Compilers removed from the A1 and A2 sandbox (2026-10-29)

D036 found that the sandbox exposed the container's compilers. GPT-6 Luna compiled and ran C++ in three R0
conversations (`frontier_openai/R0_A1_events_r3`, `R0_A2_observables_r2`, `R0_A2_observables_r5`) and in one
pilot conversation (`P0_A1_events_r5`). STUDY_PLAN Section 5.2 defines A1 and A2 as Python conditions, so a
compiled simulation falls outside the condition.

- **Change, for every run after M4-R0.** `ai/harness/sandbox.py` now also hides:
  - the GCC and LLVM back ends (`/usr/lib/gcc`, `/usr/libexec/gcc`, `/usr/lib/llvm-*`);
  - the assembler and linkers (`as`, `ld.bfd`, `ld.gold`, `nasm`), each covered by `/dev/null`;
  - Go, Java, Node and the Playwright browsers.

  The compiler drivers stay on disk but cannot build anything. The shell, its utilities, and the venv Python with
  numpy, scipy and multiprocessing are unchanged.
- **Test.** `tests/test_ai_harness.py::test_sandbox_runs_python_but_no_compiler` checks the sandbox:
  - `gcc`, `g++`, `clang`, `gfortran`, `as`, `ld`, `node`, `java` and `go` all fail inside it;
  - scipy runs.
- **M4-R0.** The three conversations stay in the data as run (Section 10.4).
  - `analysis/m5.py` finds them from the code each model ran (`compiled_conversations`). It reports every affected
    model–condition cell with and without them (`sensitivity_no_compiler` in `M5_M4-R0.json`).
  - **No breakpoint changes,** under either shape criterion (D038).
  - **GPT-6 Luna A2.** The median D/Φ₀ error moves from −10.6% to −13.2%. Its best answer (r2: dose +0.1%,
    scattered fraction −0.7%) came from compiled C++.
  - **GPT-6 Luna events.** Its only answered events call, A1 r3, is a C++ conversation. Without it, GPT-6 Luna
    has no scored events answer.
- **Not rerun.** Rerunning the three conversations under the new sandbox would cost about $0.10; their measured
  charge was $0.08. That needs Humza's approval (Section 10.5). A rerun is a new sample, so it would be reported
  beside the original, not replace it.

## D038. A second shape criterion with physical tolerances (2026-10-29; exploratory for M4-R0)

The pre-specified shape criterion (D028 item 4) passes a distribution when its W1 and JSD to the Monte Carlo are
within 3 × the Monte Carlo's half-sample distances. Those distances measure the reference's own statistical noise,
10⁻⁴ to 10⁻³ in W1 (MeV or cos θ).
- An answer passes only if it is statistically indistinguishable from a 10⁸-history reference, so an answer
  must itself be a Monte Carlo of that precision.
- Every model and condition broke at Layer 1, including answers whose spectra match the reference within a few
  percent.
- The criterion therefore measures the reference's precision, not whether an answer is physically right. That is
  a design error in D028 item 4, and it was found only after M4 was scored.

- **Revised criterion.** A distribution passes when its W1 to the Monte Carlo is within a tolerance on the scale of
  the scalar tolerances:
  - **Energy spectra:** W1 ≤ 5% of the Monte Carlo mean energy of that distribution. For a pure shift, W1 equals
    the shift of the mean, so this is the 5% scalar tolerance applied to the spectrum.
  - **cos θ distributions:** W1 ≤ 0.1, which is 5% of the range of cos θ.

  The resolution floors (the largest W1 between option4 and the other Geant4 lists, all below 0.0014) stay.
  JSD is reported but does not enter the verdict: on fine bins it penalizes noise in a model's histogram and has
  no physical unit.
- **Tolerances at R0** (`distribution_w1_tolerance_D038` in `M5_M4-R0.json`):
  - primary photon spectrum 0.063 MeV; scattered photon spectrum 0.024 MeV;
  - initial electron spectrum 0.020 MeV (0.030 MeV for electrons set in motion by primary photons);
  - electron fluence spectrum 0.022 MeV;
  - all angular distributions 0.1.
- **Controls** (`controls_revised` in `M5_controls.json`) behave as under the pre-specified criterion:
  - both other Geant4 lists show no breakpoint;
  - the +5% dose control breaks at Layer 5;
  - the uniform answer breaks at Layer 1;
  - the identity control passes everything.
- **Status.** The criterion was chosen after seeing the M4-R0 results.
  - For M4-R0 it is exploratory and is reported beside the pre-specified criterion, which remains the primary
    analysis.
  - For any later run it is the pre-registered primary shape criterion, with the constants fixed in this entry
    and in `analysis/scoring.py`.
- **Effect on M4-R0** (`models_revised`):
  - **Haiku A1 and A2** break at Layer 2 instead of Layer 1, and every Layer 1 and Layer 3 distribution passes.
    What fails is the photoelectric share of interactions (0.7% against 2.4%) and, in Layer 4, the electron
    angular distribution (W1 0.22 and 0.24). D/Φ₀ passes.
  - **Every other model and condition** still breaks at Layer 1, mostly on the scattered-photon spectrum:
    - DeepSeek A1 and A2: W1 0.032 and 0.034 MeV, against 0.024;
    - GPT-6 Luna, Qwen and Gemini are further off.
