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
