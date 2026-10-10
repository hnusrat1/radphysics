#ifndef CO60_TALLIES_HH
#define CO60_TALLIES_HH

#include "Config.hh"

#include "G4String.hh"

#include <map>
#include <string>
#include <vector>

// A named, weighted histogram with fixed dimensions, stored flat. Energy and
// length values are accumulated already converted to the output units (MeV, cm),
// so Geant4 internal units never reach the output (STUDY_PLAN Section 8).
struct Tally
{
  std::vector<int> dims;
  std::vector<double> data;
  std::string unit;
  std::string meaning;

  Tally() = default;
  Tally(std::vector<int> d, std::string u, std::string what)
    : dims(std::move(d)), unit(std::move(u)), meaning(std::move(what))
  {
    size_t n = 1;
    for (int x : dims) n *= static_cast<size_t>(x);
    data.assign(n, 0.);
  }
  double& at(std::initializer_list<int> idx)
  {
    size_t flat = 0;
    size_t k = 0;
    for (int i : idx) flat = flat * static_cast<size_t>(dims[k++]) + static_cast<size_t>(i);
    return data[flat];
  }
};

// One record of the sampled-events task (STUDY_PLAN Section 4.1).
struct EventRecord
{
  long eventId;
  double weight;
  int photonCat;
  int process;              // index into kProcessNames
  double eIn;               // MeV
  double cosIn;             // incident direction relative to the beam axis
  double eOut;              // scattered photon energy (MeV), -1 if none
  double cosScatter;        // cos of angle between incident and scattered photon, NaN if none
  int nCharged;
  double tMax;              // largest charged-secondary kinetic energy (MeV)
  int pdgMax;               // particle with tMax (11 or -11)
  double cosElectron;       // cos of angle between incident photon and that particle
  double tChargedSum;       // all charged secondaries (MeV)
  double ePositron;         // positron kinetic energy for pair production (MeV), -1 otherwise
  double eSecPhotons;       // fluorescence photons created in the interaction (MeV)
  double eLocal;            // energy deposited locally at the interaction (MeV)
  double x, y, z;           // cm, z is depth below the phantom surface
};

enum ProcessIndex : int { kCompt = 0, kPhot = 1, kConv = 2, kRayl = 3, kOtherProc = 4, kNProc = 5 };
extern const char* kProcessNames[kNProc];
int ProcessIndexOf(const G4String& name);

class Tallies
{
 public:
  explicit Tallies(const Config& c);
  void Merge(const Tallies& other);
  void WriteJson(std::ostream& os) const;

  std::map<std::string, Tally> t;    // all histograms by name
  std::map<std::string, double> s;   // scalar sums by name
  double energyBalanceMaxAbs = 0.;   // MeV, maximum over photon interactions in the scoring volume

  // History-by-history variance: per-event sums, folded into s["<name>"] and
  // s["<name>_sq"] at the end of each event.
  static constexpr const char* kPerEvent[4] = {"edep", "kerma_energy_transfer", "kcol_radiative_loss",
                                               "photon_track_length_total"};
  double current[4] = {0., 0., 0., 0.};
  void EndOfEvent();
  std::vector<EventRecord> events;
  int eventSampleMax = 0;

  Tally& operator[](const std::string& name) { return t.at(name); }
};

#endif
