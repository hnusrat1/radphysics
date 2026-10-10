#include "ImportanceWorld.hh"

#include "Config.hh"

#include "G4Exception.hh"
#include "G4GeometryCell.hh"
#include "G4IStore.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"

ImportanceWorld* ImportanceWorld::fInstance = nullptr;
G4VPhysicalVolume* ImportanceWorld::fGhost = nullptr;

ImportanceWorld::ImportanceWorld(const G4String& name) : G4VUserParallelWorld(name)
{
  fInstance = this;
}

void ImportanceWorld::Construct()
{
  const Config& c = Config::Get();
  fGhost = GetWorld();
  G4LogicalVolume* mother = fGhost->GetLogicalVolume();
  G4ThreeVector motherCentre;  // global position of the mother's origin
  G4int copy = 1;
  for (const auto& cell : c.importanceCells) {
    const G4double halfZ = 0.5 * (cell.zmax - cell.zmin);
    const G4double zc = 0.5 * (cell.zmax + cell.zmin);
    auto* solid = new G4Tubs("ImpCell" + std::to_string(copy), 0., cell.radius, halfZ, 0., CLHEP::twopi);
    auto* logical = new G4LogicalVolume(solid, nullptr, "ImpCell" + std::to_string(copy));
    fCells.push_back(new G4PVPlacement(nullptr, G4ThreeVector(0, 0, zc) - motherCentre, logical,
                                       "ImpCell" + std::to_string(copy), mother, false, copy, true));
    mother = logical;
    motherCentre = G4ThreeVector(0, 0, zc);
    ++copy;
  }
}

void ImportanceWorld::CreateImportanceStore()
{
  const Config& c = Config::Get();
  G4IStore* store = G4IStore::GetInstance(GetName());
  store->AddImportanceGeometryCell(1., *fGhost, 0);
  for (size_t i = 0; i < fCells.size(); ++i) {
    store->AddImportanceGeometryCell(c.importanceCells[i].importance, *fCells[i], static_cast<G4int>(i + 1));
    G4cout << "Importance cell " << i + 1 << ": r <= " << c.importanceCells[i].radius / cm << " cm, "
           << c.importanceCells[i].zmin / cm << " <= z <= " << c.importanceCells[i].zmax / cm
           << " cm, importance " << c.importanceCells[i].importance << G4endl;
  }
}
