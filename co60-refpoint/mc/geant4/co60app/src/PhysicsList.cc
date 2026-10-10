#include "PhysicsList.hh"

#include "Config.hh"

#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4Exception.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4RToEConvForElectron.hh"
#include "G4RToEConvForGamma.hh"
#include "G4RToEConvForPositron.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <memory>

PhysicsList::PhysicsList(const G4String& emName)
{
  SetVerboseLevel(1);
  if (emName == "option4") RegisterPhysics(new G4EmStandardPhysics_option4());
  else if (emName == "livermore") RegisterPhysics(new G4EmLivermorePhysics());
  else if (emName == "penelope") RegisterPhysics(new G4EmPenelopePhysics());
  else {
    G4Exception("PhysicsList", "CO60-PHYS", FatalErrorInArgument,
                ("unknown EM physics '" + emName + "'; use option4, livermore or penelope").c_str());
  }
}

G4double PhysicsList::RangeForThreshold(const G4String& particle, G4double target, G4bool atMost)
{
  const G4Material* water = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  std::unique_ptr<G4VRangeToEnergyConverter> conv;
  if (particle == "e-") conv = std::make_unique<G4RToEConvForElectron>();
  else if (particle == "e+") conv = std::make_unique<G4RToEConvForPositron>();
  else conv = std::make_unique<G4RToEConvForGamma>();

  // Energy threshold increases monotonically with range cut; bisect in log(range).
  G4double lo = std::log(1e-4 * mm), hi = std::log(10. * mm);
  for (int i = 0; i < 60; ++i) {
    const G4double mid = 0.5 * (lo + hi);
    if (conv->Convert(std::exp(mid), water) < target) lo = mid;
    else hi = mid;
  }
  // hi: threshold just above target; lo: just below.
  return std::exp(atMost ? lo : hi);
}

void PhysicsList::SetCuts()
{
  const Config& c = Config::Get();
  if (c.tableMode) {
    SetDefaultCutValue(0.7 * mm);
    G4VUserPhysicsList::SetCuts();
    return;
  }
  const G4double rElectron = RangeForThreshold("e-", c.electronThreshold);
  const G4double rPositron = RangeForThreshold("e+", c.electronThreshold);
  // The gamma threshold only needs to stay at or below its bound; capping the
  // gamma range at 1 mm keeps it at a few keV in water.
  const G4double rGamma = std::min(RangeForThreshold("gamma", c.gammaThreshold, true), 1. * mm);
  SetCutValue(rGamma, "gamma");
  SetCutValue(rElectron, "e-");
  SetCutValue(rPositron, "e+");
  SetCutValue(rElectron, "proton");
  if (verboseLevel > 0) DumpCutValuesTable();
}
