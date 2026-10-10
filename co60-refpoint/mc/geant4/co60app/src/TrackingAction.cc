#include "Actions.hh"
#include "Config.hh"
#include "Tallies.hh"
#include "TrackInfo.hh"

#include "G4Gamma.hh"
#include "G4Track.hh"
#include "G4TrackingManager.hh"

#include <cmath>

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
  if (track->GetUserInformation() != nullptr || Config::Get().tableMode) return;
  auto* info = new TrackInfo();
  if (track->GetParentID() == 0 && track->GetDefinition() == G4Gamma::Definition()) {
    const Config& c = Config::Get();
    const double e = track->GetKineticEnergy();
    info->photonCat = std::abs(e - c.line1) < std::abs(e - c.line2) ? kPrimaryLine1 : kPrimaryLine2;
    fRun->GetTallies()->s[info->photonCat == kPrimaryLine1 ? "histories_line1" : "histories_line2"] += 1.;
  }
  else {
    // Every secondary should already carry information from the step that
    // created it; count the exceptions.
    fRun->GetTallies()->s["untagged_tracks"] += 1.;
  }
  fpTrackingManager->SetUserTrackInformation(info);
}
