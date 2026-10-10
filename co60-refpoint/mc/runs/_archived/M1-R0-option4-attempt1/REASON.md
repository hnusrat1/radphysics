# Archived: M1-R0-option4, first launch (2026-10-10)

Stopped after batch 0 of 10. The interrupted batch 1 had produced no output and was removed. Batch 0
(1.2 x 10^7 histories) is kept, and it showed two problems that the production configuration had to fix first:

1. **Rayleigh scattering missing above 1.022 MeV.** G4EmStandardPhysics_option4 enables G4GammaGeneralProcess. That
   process's energy zone from 2 m_e c^2 to 100 MeV selects pair production, Compton, photoelectric and photonuclear
   only (G4GammaGeneralProcess.cc, PostStepDoIt, case 2), so no primary photon at either Co-60 line could undergo
   coherent scattering. Layer 2 showed zero Rayleigh events for primaries where about 30 were expected per batch.
   Fixed by `/process/em/UseGeneralProcess false` (DECISIONS D019).
2. **Low-energy electron fluence distorted by the linear-loss estimator.** The Layer 4 estimator spreads each step's
   track length uniformly over the step's energy loss (linear energy loss, STUDY_PLAN Layer 4). Geant4 transports an
   electron below about 20 keV to rest in one step (step function final range 10 um), and over such a step the path
   per unit energy goes as 1/L(E), not uniformly. The fluence-to-dose closure, the integral of fluence times
   restricted stopping power divided by the scored dose, came out at 1.064 instead of about 1. A second estimator
   that spreads track length in proportion to 1/L(E) was added; both are now scored (DECISIONS D020).

Batch 0 results remain valid for everything else and are kept for comparison.
