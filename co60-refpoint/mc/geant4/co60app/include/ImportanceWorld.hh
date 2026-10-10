#ifndef CO60_IMPORTANCEWORLD_HH
#define CO60_IMPORTANCEWORLD_HH

#include "G4VUserParallelWorld.hh"

#include <vector>

class G4VPhysicalVolume;

// Parallel world of nested cylinders on the beam axis used for photon
// importance sampling (G4ImportanceBiasing). A photon entering a cell of higher
// importance is split, one entering a cell of lower importance is Russian-
// rouletted; weights keep every tally unbiased. Cells carry no material and do
// not affect the mass geometry or the physics.
class ImportanceWorld : public G4VUserParallelWorld
{
 public:
  explicit ImportanceWorld(const G4String& name);
  void Construct() override;
  // Fill the G4IStore after /run/initialize (UI command /co60/vr/applyImportance).
  void CreateImportanceStore();

  static ImportanceWorld* Instance() { return fInstance; }
  static G4VPhysicalVolume* fGhost;

 private:
  static ImportanceWorld* fInstance;
  std::vector<G4VPhysicalVolume*> fCells;
};

#endif
