#include "Tallies.hh"

#include "TrackInfo.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>

const char* kProcessNames[kNProc] = {"compt", "phot", "conv", "Rayl", "other"};

int ProcessIndexOf(const G4String& name)
{
  for (int i = 0; i < kOtherProc; ++i) {
    if (name == kProcessNames[i]) return i;
  }
  return kOtherProc;
}

Tallies::Tallies(const Config& c)
{
  // Energy dimensions carry an underflow bin (index 0) and an overflow bin (n + 1).
  const int nPh = c.photonE.n + 2;
  const int nEl = c.electronE.n + 2;
  const int nCos = c.cosTheta.n;
  const int nDepth = c.depth.n + 2;

  // Layer 1: photons arriving at the point (track-length estimator, sum of w * L)
  t["photon_fluence"] = Tally({kNPhotonCat, nPh}, "cm", "sum w*L by photon category and energy bin");
  t["photon_energy_fluence"] = Tally({kNPhotonCat}, "MeV cm", "sum w*L*E by photon category");
  t["photon_track_length"] = Tally({kNPhotonCat}, "cm", "sum w*L by photon category");
  t["photon_angular"] = Tally({kNPhotonCat, nCos}, "cm", "sum w*L by photon category and cos(theta) to beam axis");

  // Layer 2: photon interactions in the scoring volume
  t["interactions"] = Tally({kNPhotonCat, kNProc}, "1", "sum w by incident photon category and process");
  t["interaction_energy_transfer"] = Tally({kNPhotonCat, kNProc}, "MeV", "sum w*E_tr by category and process");
  t["interaction_incident_energy"] = Tally({kNPhotonCat, kNProc}, "MeV", "sum w*E_in by category and process");
  t["interaction_spectrum"] = Tally({kNProc, nPh}, "1", "sum w by process and incident photon energy bin");

  // Layer 3: charged particles set in motion by photon interactions in the scoring volume
  t["initial_spectrum"] =
    Tally({kNProc, kNPhotonCat, 2, nEl}, "1", "sum w by process, parent photon category, particle (e-, e+), T bin");
  t["initial_count"] = Tally({kNProc, kNPhotonCat, 2}, "1", "sum w by process, parent category, particle");
  t["initial_energy"] = Tally({kNProc, kNPhotonCat, 2}, "MeV", "sum w*T by process, parent category, particle");
  t["initial_cos"] =
    Tally({kNProc, 2, nCos}, "1", "sum w by process, parent primary(0)/non-primary(1), cos(theta) to beam axis");

  // Layer 4: electrons slowing down (track-length estimator, linear energy loss along each step)
  t["electron_fluence"] = Tally({nEl}, "cm", "e-, all, sum w*L by energy bin");
  t["electron_fluence_delta"] = Tally({nEl}, "cm", "e- created by charged particles (delta rays and descendants)");
  t["electron_fluence_by_parent"] =
    Tally({3, nEl}, "cm", "e- by parent photon: primary(0), non-primary(1), unknown(2)");
  t["positron_fluence"] = Tally({nEl}, "cm", "e+, sum w*L by energy bin");
  // Same quantities with each step's track length spread over energy as 1/L(E)
  // (restricted CSDA) instead of uniformly (DECISIONS D020).
  t["electron_fluence_csda"] = Tally({nEl}, "cm", "e-, all, CSDA-weighted track length by energy bin");
  t["electron_fluence_delta_csda"] = Tally({nEl}, "cm", "e- delta rays and descendants, CSDA-weighted");
  t["electron_fluence_by_parent_csda"] =
    Tally({3, nEl}, "cm", "e- by parent photon (primary, non-primary, unknown), CSDA-weighted");
  t["positron_fluence_csda"] = Tally({nEl}, "cm", "e+, CSDA-weighted track length by energy bin");
  t["electron_angular"] = Tally({nCos}, "cm", "e-, sum w*L by cos(theta) to beam axis (pre-step direction)");
  t["positron_angular"] = Tally({nCos}, "cm", "e+, sum w*L by cos(theta)");
  t["charged_energy_loss"] =
    Tally({2, 2}, "MeV", "energy lost in the scoring volume by particle (e-, e+) and type (collision, radiative)");

  // Layer 5: energy deposited
  t["edep_by_parent"] =
    Tally({kNPhotonCat + 1, 3}, "MeV",
          "energy deposited in the scoring volume by parent photon category (index 4: unknown) and depositing "
          "particle (gamma, e-, e+)");
  t["depth_edep"] = Tally({nDepth}, "MeV", "central-axis depth scan, energy deposited within depthRadius");

  for (const char* k : {"histories_line1", "histories_line2", "rr_killed", "rr_survived", "untagged_tracks",
                        "interactions_unweighted"})
  {
    s[k] = 0.;
  }
  for (const char* k : kPerEvent) {
    s[k] = 0.;
    s[std::string(k) + "_sq"] = 0.;
  }
  eventSampleMax = c.eventSampleMax;
}

void Tallies::EndOfEvent()
{
  for (int i = 0; i < 4; ++i) {
    s[kPerEvent[i]] += current[i];
    s[std::string(kPerEvent[i]) + "_sq"] += current[i] * current[i];
    current[i] = 0.;
  }
}

void Tallies::Merge(const Tallies& o)
{
  for (auto& [name, tally] : t) {
    const auto& src = o.t.at(name).data;
    for (size_t i = 0; i < tally.data.size(); ++i) tally.data[i] += src[i];
  }
  for (auto& [name, v] : s) v += o.s.at(name);
  energyBalanceMaxAbs = std::max(energyBalanceMaxAbs, o.energyBalanceMaxAbs);
  for (const auto& e : o.events) {
    if (static_cast<int>(events.size()) < eventSampleMax) events.push_back(e);
  }
}

void Tallies::WriteJson(std::ostream& os) const
{
  os << std::setprecision(std::numeric_limits<double>::max_digits10);
  os << "  \"scalars\": {\n";
  size_t i = 0;
  for (const auto& [name, v] : s) {
    os << "    \"" << name << "\": " << v << (++i < s.size() ? ",\n" : "\n");
  }
  os << "  },\n  \"energy_balance_max_abs_MeV\": " << energyBalanceMaxAbs << ",\n";
  os << "  \"event_sample_records\": " << events.size() << ",\n";
  os << "  \"tallies\": {\n";
  i = 0;
  for (const auto& [name, tally] : t) {
    os << "    \"" << name << "\": {\"unit\": \"" << tally.unit << "\", \"meaning\": \"" << tally.meaning
       << "\", \"dims\": [";
    for (size_t k = 0; k < tally.dims.size(); ++k) os << (k ? ", " : "") << tally.dims[k];
    os << "], \"data\": [";
    for (size_t k = 0; k < tally.data.size(); ++k) os << (k ? ", " : "") << tally.data[k];
    os << "]}" << (++i < t.size() ? ",\n" : "\n");
  }
  os << "  }\n";
}
