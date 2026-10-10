#ifndef CO60_ACTIONS_HH
#define CO60_ACTIONS_HH

#include "G4String.hh"
#include "G4Timer.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VUserActionInitialization.hh"

class Tallies;

class RunAction : public G4UserRunAction
{
 public:
  G4Run* GenerateRun() override;
  void BeginOfRunAction(const G4Run*) override;
  void EndOfRunAction(const G4Run*) override;
  Tallies* GetTallies() { return fTallies; }

  static G4String gPhysicsName;  // set by main

 private:
  void CheckRoi() const;
  void WriteOutput(const G4Run* run, const Tallies& t) const;
  void WriteEventSample(const Tallies& t) const;
  void WriteTables() const;

  Tallies* fTallies = nullptr;
  G4Timer fTimer;
};

// Assigns TrackInfo to primary photons; secondaries get theirs in SteppingAction
// at the step that created them.
class TrackingAction : public G4UserTrackingAction
{
 public:
  explicit TrackingAction(RunAction* r) : fRun(r) {}
  void PreUserTrackingAction(const G4Track*) override;

 private:
  RunAction* fRun;
};

// Russian roulette of electrons born outside the region of interest. Unbiased:
// survivors carry weight 1/p. Positrons, photons, and electrons tagged for
// collision kerma are never rouletted.
class StackingAction : public G4UserStackingAction
{
 public:
  explicit StackingAction(RunAction* r) : fRun(r) {}
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track*) override;

 private:
  RunAction* fRun;
};

class SteppingAction : public G4UserSteppingAction
{
 public:
  explicit SteppingAction(RunAction* r) : fRun(r) {}
  void UserSteppingAction(const G4Step*) override;

 private:
  RunAction* fRun;
};

// Folds per-event sums into the history-by-history variance accumulators.
class EventAction : public G4UserEventAction
{
 public:
  explicit EventAction(RunAction* r) : fRun(r) {}
  void EndOfEventAction(const G4Event*) override;

 private:
  RunAction* fRun;
};

class ActionInitialization : public G4VUserActionInitialization
{
 public:
  void BuildForMaster() const override;
  void Build() const override;
};

#endif
