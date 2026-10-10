#include "DetectorConstruction.hh"

#include "Config.hh"

#include "G4Box.hh"
#include "G4Exception.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Tubs.hh"

#include <algorithm>

const G4VPhysicalVolume* DetectorConstruction::fPhantomPV = nullptr;
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

  const G4double half = 0.5 * c.phantomSize;
  if (c.scoringDepth - 0.5 * c.scoringThickness < 0. || c.scoringDepth + 0.5 * c.scoringThickness > c.phantomSize
      || c.scoringRadius > half)
  {
    G4Exception("DetectorConstruction", "CO60-GEOM", FatalException, "scoring cylinder outside the phantom");
  }

  // World: the source sits at z = -SSD; leave 10 cm of vacuum around everything.
  const G4double worldHalfXY = std::max(half, 0.5 * std::max(c.fieldX, c.fieldY)) + 10 * cm;
  const G4double worldHalfZ = std::max(c.ssd, c.phantomSize) + 10 * cm;
  auto* worldS = new G4Box("World", worldHalfXY, worldHalfXY, worldHalfZ);
  auto* worldL = new G4LogicalVolume(worldS, vacuum, "World");
  auto* worldP = new G4PVPlacement(nullptr, {}, worldL, "World", nullptr, false, 0, true);

  auto* phantomS = new G4Box("Phantom", half, half, half);
  auto* phantomL = new G4LogicalVolume(phantomS, water, "Phantom");
  fPhantomPV = new G4PVPlacement(nullptr, {0, 0, half}, phantomL, "Phantom", worldL, false, 0, true);

  auto* scoringS = new G4Tubs("Scoring", 0., c.scoringRadius, 0.5 * c.scoringThickness, 0., CLHEP::twopi);
  auto* scoringL = new G4LogicalVolume(scoringS, water, "Scoring");
  fScoringPV =
    new G4PVPlacement(nullptr, {0, 0, c.scoringDepth - half}, scoringL, "Scoring", phantomL, false, 0, true);
  fScoringMass = scoringL->GetMass();

  return worldP;
}
