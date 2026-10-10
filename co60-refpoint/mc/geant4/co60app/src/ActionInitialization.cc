#include "Actions.hh"
#include "PrimaryGeneratorAction.hh"
#include "Tallies.hh"

void EventAction::EndOfEventAction(const G4Event*)
{
  fRun->GetTallies()->EndOfEvent();
}

void ActionInitialization::BuildForMaster() const
{
  SetUserAction(new RunAction());
}

void ActionInitialization::Build() const
{
  auto* run = new RunAction();
  SetUserAction(run);
  SetUserAction(new PrimaryGeneratorAction());
  SetUserAction(new EventAction(run));
  SetUserAction(new TrackingAction(run));
  SetUserAction(new StackingAction(run));
  SetUserAction(new SteppingAction(run));
}
