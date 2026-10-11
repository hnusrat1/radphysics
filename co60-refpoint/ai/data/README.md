# Data files for condition A2

Condition A2 gives the model the NIST XCOM and ESTAR tables for liquid water as files (STUDY_PLAN Section 5.2).
physics.nist.gov could not be reached from the build environment, so the files here are reproductions
(DECISIONS D028 item 10):

| File | Built from | Method |
|---|---|---|
| `xcom_water.txt` | XCOM elemental cross sections distributed with EGSnrc v2026 (`HEN_HOUSE/data/xcom_*.data`) | mass-fraction mixture, as XCOM does for compounds |
| `estar_water.txt` | ESTAR density-effect corrections (`water_liquid.density`) and Seltzer–Berger bremsstrahlung tables (`nist_brems.data`), both distributed with EGSnrc | Bethe formula with I = 75 eV; radiative stopping power integrated from the scaled cross sections; CSDA range and radiation yield integrated from 1 keV |

Build: `python -m ai.data.build_tables`. Check: `python -m ai.data.validate_tables`. Below 1.4 MeV the tables agree with
Geant4 option4 to within 0.1% for photon coefficients above 30 keV, and to within 0.1% for radiative stopping
power. Collision stopping power agrees within 0.6% above 0.5 MeV, with I = 75 eV here against 78 eV in G4_WATER.

**Before M4 (Humza):** if you can reach NIST, download the genuine tables and replace these two files under the same
names, keeping the column order given in each header:
- XCOM (https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html): compound H2O, by mass H 0.111894, O 0.888106;
  standard grid 1 keV to 20 MeV; all partial coefficients and both totals, cm²/g.
- ESTAR (https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html): material "WATER, LIQUID", default energies
  from 0.01 MeV to 20 MeV.

Then rerun `validate_tables` and `python -m ai.harness.build_prompts --version v1`. The prompts describe the files
but do not include them, so the prompt hashes do not change. Record the swap in DECISIONS.
