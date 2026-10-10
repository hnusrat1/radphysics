#ifndef CO60_PRIMARYGENERATORACTION_HH
#define CO60_PRIMARYGENERATORACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4ParticleGun;

// Source model A (STUDY_PLAN Section 2): point source at z = -SSD emitting
// photons uniformly in solid angle within the pyramid that defines the field at
// the phantom surface, with the two Co-60 lines.
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
 public:
  PrimaryGeneratorAction();
  ~PrimaryGeneratorAction() override;
  void GeneratePrimaries(G4Event*) override;

  // Solid angle of the emission pyramid (sr) and primary fluence per emitted
  // photon at distance ssd + depth on the axis, without the phantom (1/area).
  static G4double PyramidSolidAngle();
  static G4double FluenceOnAxis(G4double depth);

 private:
  G4ParticleGun* fGun;
};

#endif
