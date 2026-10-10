#ifndef CO60_PHYSICSLIST_HH
#define CO60_PHYSICSLIST_HH

#include "G4VModularPhysicsList.hh"

// Electromagnetic physics only: one of the three study constructors
// (G4EmStandardPhysics_option4, G4EmLivermorePhysics, G4EmPenelopePhysics),
// used unmodified. Production cuts are given as energy thresholds in G4_WATER
// and converted to the range cut Geant4 needs.
class PhysicsList : public G4VModularPhysicsList
{
 public:
  explicit PhysicsList(const G4String& emName);
  void SetCuts() override;

  // Range cut whose energy threshold in G4_WATER equals target (to bisection
  // precision), from above, or from below when atMost is true.
  static G4double RangeForThreshold(const G4String& particle, G4double target, G4bool atMost = false);
};

#endif
