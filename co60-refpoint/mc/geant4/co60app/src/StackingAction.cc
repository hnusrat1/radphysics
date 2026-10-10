#include "Actions.hh"
#include "Config.hh"
#include "Tallies.hh"
#include "TrackInfo.hh"

#include "G4Electron.hh"
#include "G4Track.hh"
#include "Randomize.hh"

#include <cmath>

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* track)
{
  const Config& c = Config::Get();
  if (!c.rrEnabled || track->GetParentID() == 0 || track->GetDefinition() != G4Electron::Definition()) {
    return fUrgent;
  }
  const auto* info = static_cast<const TrackInfo*>(track->GetUserInformation());
  if (info != nullptr && info->kcolTag) return fUrgent;
  const G4ThreeVector& x = track->GetPosition();
  if (c.InRoi(std::hypot(x.x(), x.y()), x.z())) return fUrgent;

  Tallies* t = fRun->GetTallies();
  if (G4UniformRand() < c.rrSurvival) {
    const_cast<G4Track*>(track)->SetWeight(track->GetWeight() / c.rrSurvival);
    t->s["rr_survived"] += 1.;
    return fUrgent;
  }
  t->s["rr_killed"] += 1.;
  return fKill;
}
