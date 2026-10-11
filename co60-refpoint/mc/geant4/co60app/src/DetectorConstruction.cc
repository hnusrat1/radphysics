#include "DetectorConstruction.hh"

#include "Config.hh"

#include "G4Box.hh"
#include "G4Exception.hh"
#include "G4IStore.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Tubs.hh"

#include <algorithm>

const G4VPhysicalVolume* DetectorConstruction::fWorldPV = nullptr;
const G4VPhysicalVolume* DetectorConstruction::fPhantomPV = nullptr;
std::vector<const G4VPhysicalVolume*> DetectorConstruction::fCellPVs;
const G4VPhysicalVolume* DetectorConstruction::fScoringPV = nullptr;
G4double DetectorConstruction::fScoringMass = 0.;

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  const Config& c = Config::Get();
  auto* nist = G4NistManager::Instance();
  G4Material* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
  G4Material* water = nist->FindOrBuildMaterial("G4_WATER");

  if (c.tableMode) {
    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* worldS = new G4Box("World", 50 * cm, 50 * cm, 50 * cm);
    auto* worldL = new G4LogicalVolume(worldS, vacuum, "World");
    auto* worldP = new G4PVPlacement(nullptr, {}, worldL, "World", nullptr, false, 0, true);
    auto* slabS = new G4Box("Slab", 10 * cm, 10 * cm, 10 * cm);
    new G4PVPlacement(nullptr, {0, 0, -20 * cm}, new G4LogicalVolume(slabS, water, "WaterSlab"), "WaterSlab",
                      worldL, false, 0, true);
    new G4PVPlacement(nullptr, {0, 0, 20 * cm}, new G4LogicalVolume(slabS, air, "AirSlab"), "AirSlab", worldL,
                      false, 0, true);
    return worldP;
  }

  const G4double half = c.PhantomHalfXY();
  const G4double depth = c.PhantomDepth();
  if (c.scoringDepth - 0.5 * c.scoringThickness < 0. || c.scoringDepth + 0.5 * c.scoringThickness > depth
      || c.scoringRadius > half)
  {
    G4Exception("DetectorConstruction", "CO60-GEOM", FatalException, "scoring cylinder outside the phantom");
  }

  // World: the source sits at z = -SSD; leave 10 cm of vacuum around everything.
  const G4double worldHalfXY = std::max({half, 0.5 * std::max(c.fieldX, c.fieldY), c.fieldRadius}) + 10 * cm;
  const G4double worldHalfZ = std::max(c.ssd, depth) + 10 * cm;
  auto* worldS = new G4Box("World", worldHalfXY, worldHalfXY, worldHalfZ);
  auto* worldL = new G4LogicalVolume(worldS, vacuum, "World");
  auto* worldP = new G4PVPlacement(nullptr, {}, worldL, "World", nullptr, false, 0, true);

  G4VSolid* phantomS = nullptr;
  if (c.cylinderPhantom) phantomS = new G4Tubs("Phantom", 0., half, 0.5 * depth, 0., CLHEP::twopi);
  else phantomS = new G4Box("Phantom", half, half, half);
  auto* phantomL = new G4LogicalVolume(phantomS, water, "Phantom");
  fPhantomPV = new G4PVPlacement(nullptr, {0, 0, 0.5 * depth}, phantomL, "Phantom", worldL, false, 0, true);

  fWorldPV = worldP;

  // Mass-geometry importance cells: nested water cylinders, clipped to the phantom.
  // The scoring cylinder must lie inside the innermost cell.
  G4LogicalVolume* mother = phantomL;
  G4double motherZ = 0.5 * depth;  // global z of the mother's origin
  fCellPVs.clear();
  if (c.importanceEnabled && c.importanceMassWorld) {
    G4int copy = 1;
    for (const auto& cell : c.importanceCells) {
      const G4double z0 = std::max(cell.zmin, 0.);
      const G4double z1 = std::min(cell.zmax, depth);
      const G4double r = std::min(cell.radius, half);
      const G4String name = "ImpCell" + std::to_string(copy);
      auto* solid = new G4Tubs(name, 0., r, 0.5 * (z1 - z0), 0., CLHEP::twopi);
      auto* logical = new G4LogicalVolume(solid, water, name);
      fCellPVs.push_back(new G4PVPlacement(nullptr, {0, 0, 0.5 * (z0 + z1) - motherZ}, logical, name, mother,
                                           false, copy, true));
      mother = logical;
      motherZ = 0.5 * (z0 + z1);
      ++copy;
    }
    if (!c.importanceCells.empty()) {
      const auto& in = c.importanceCells.back();
      if (c.scoringRadius > in.radius || c.scoringDepth - 0.5 * c.scoringThickness < std::max(in.zmin, 0.)
          || c.scoringDepth + 0.5 * c.scoringThickness > in.zmax)
      {
        G4Exception("DetectorConstruction", "CO60-IMP", FatalException,
                    "--importance-mass needs the scoring cylinder inside the innermost importance cell");
      }
    }
  }

  auto* scoringS = new G4Tubs("Scoring", 0., c.scoringRadius, 0.5 * c.scoringThickness, 0., CLHEP::twopi);
  auto* scoringL = new G4LogicalVolume(scoringS, water, "Scoring");
  fScoringPV =
    new G4PVPlacement(nullptr, {0, 0, c.scoringDepth - motherZ}, scoringL, "Scoring", mother, false, 0, true);
  fScoringMass = scoringL->GetMass();

  return worldP;
}

void DetectorConstruction::CreateMassImportanceStore()
{
  const Config& c = Config::Get();
  G4IStore* store = G4IStore::GetInstance();
  store->AddImportanceGeometryCell(1., *fWorldPV, 0);
  store->AddImportanceGeometryCell(1., *fPhantomPV, 0);
  G4double inner = 1.;
  for (size_t i = 0; i < fCellPVs.size(); ++i) {
    inner = c.importanceCells[i].importance;
    store->AddImportanceGeometryCell(inner, *fCellPVs[i], static_cast<G4int>(i + 1));
    G4cout << "Importance cell " << i + 1 << " (mass geometry): r <= " << c.importanceCells[i].radius / cm
           << " cm, " << std::max(c.importanceCells[i].zmin, 0.) / cm << " <= z <= "
           << c.importanceCells[i].zmax / cm << " cm, importance " << inner << G4endl;
  }
  store->AddImportanceGeometryCell(inner, *fScoringPV, 0);
}
