#ifndef CO60_DETECTORCONSTRUCTION_HH
#define CO60_DETECTORCONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"

class G4VPhysicalVolume;

// R0 geometry (STUDY_PLAN Section 2): vacuum world, liquid water cube with its
// beam-side surface at z = 0, scoring cylinder on the beam axis. The scoring
// cylinder is a G4_WATER daughter of the phantom, so the only effect of the
// daughter is a geometric boundary for step limitation.
//
// Table mode builds separate water and air slabs so that G4EmCalculator can
// evaluate stopping powers in both materials with the production physics.
class DetectorConstruction : public G4VUserDetectorConstruction
{
 public:
  G4VPhysicalVolume* Construct() override;

  static const G4VPhysicalVolume* PhantomPV() { return fPhantomPV; }
  static const G4VPhysicalVolume* ScoringPV() { return fScoringPV; }
  static G4double ScoringMass() { return fScoringMass; }  // internal units

 private:
  static const G4VPhysicalVolume* fPhantomPV;
  static const G4VPhysicalVolume* fScoringPV;
  static G4double fScoringMass;
};

#endif
