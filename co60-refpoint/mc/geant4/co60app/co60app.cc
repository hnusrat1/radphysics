// co60app: photon-to-dose scoring at the Co-60 reference point (STUDY_PLAN R0).
//
// Usage: co60app <macro> <option4|livermore|penelope> [threads] [--importance]

#include "Actions.hh"
#include "Config.hh"
#include "ImportanceWorld.hh"
#include "DetectorConstruction.hh"
#include "Messenger.hh"
#include "PhysicsList.hh"

#include "G4GeometrySampler.hh"
#include "G4ImportanceBiasing.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIcommand.hh"

#include <memory>

int main(int argc, char** argv)
{
  if (argc < 3) {
    G4cerr << "usage: co60app <macro> <option4|livermore|penelope> [threads] [--importance]" << G4endl;
    return 1;
  }
  const G4bool importance = argc > 4 && G4String(argv[4]) == "--importance";
  Config::Get().importanceEnabled = importance;
  auto messenger = std::make_unique<Messenger>();
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
  if (argc > 3) runManager->SetNumberOfThreads(G4UIcommand::ConvertToInt(argv[3]));

  RunAction::gPhysicsName = argv[2];
  auto* detector = new DetectorConstruction();
  auto* physics = new PhysicsList(argv[2]);
  // Photon importance sampling in a parallel world (unbiased; see ImportanceWorld.hh).
  const G4String parallelName = "ImportanceWorld";
  std::unique_ptr<G4GeometrySampler> sampler;
  if (importance) {
    auto* world = new ImportanceWorld(parallelName);
    detector->RegisterParallelWorld(world);
    sampler = std::make_unique<G4GeometrySampler>(parallelName, "gamma");
    sampler->SetParallel(true);
    physics->RegisterPhysics(new G4ImportanceBiasing(sampler.get(), parallelName));
  }
  runManager->SetUserInitialization(detector);
  runManager->SetUserInitialization(physics);
  runManager->SetUserInitialization(new ActionInitialization());

  G4UImanager::GetUIpointer()->ApplyCommand(G4String("/control/execute ") + argv[1]);

  delete runManager;
  return 0;
}
