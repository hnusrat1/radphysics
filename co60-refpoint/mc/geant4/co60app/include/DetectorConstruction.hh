#ifndef CO60_DETECTORCONSTRUCTION_HH
#define CO60_DETECTORCONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"

#include <vector>

class G4VPhysicalVolume;

// R0 geometry (STUDY_PLAN Section 2): vacuum world, liquid water cube with its
// beam-side surface at z = 0, scoring cylinder on the beam axis. The scoring
// cylinder is a G4_WATER daughter of the phantom, so the only effect of the
// daughter is a geometric boundary for step limitation. With --importance-mass
// the importance cells are nested G4_WATER cylinders between the phantom and the
// scoring cylinder, clipped to the phantom; again only boundaries are added.
//
// Table mode builds separate water and air slabs so that G4EmCalculator can
// evaluate stopping powers in both materials with the production physics.
class DetectorConstruction : public G4VUserDetectorConstruction
{
 public:
  G4VPhysicalVolume* Construct() override;

  static const G4VPhysicalVolume* WorldPV() { return fWorldPV; }
  static const G4VPhysicalVolume* PhantomPV() { return fPhantomPV; }
  static const G4VPhysicalVolume* ScoringPV() { return fScoringPV; }
  static G4double ScoringMass() { return fScoringMass; }  // internal units

  // Mass-geometry importance sampling (--importance-mass): fill the G4IStore after
  // /run/initialize (UI command /co60/vr/applyImportance).
  static void CreateMassImportanceStore();

 private:
  static const G4VPhysicalVolume* fWorldPV;
  static const G4VPhysicalVolume* fPhantomPV;
  static const G4VPhysicalVolume* fScoringPV;
  static std::vector<const G4VPhysicalVolume*> fCellPVs;
  static G4double fScoringMass;
};

#endif
