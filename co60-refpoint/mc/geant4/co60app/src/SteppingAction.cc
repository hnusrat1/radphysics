// Scoring for all five layers (STUDY_PLAN Section 3).
//
// Units: every value added to a tally is converted here to MeV or cm.

#include "Actions.hh"
#include "Config.hh"
#include "DetectorConstruction.hh"
#include "Tallies.hh"
#include "TrackInfo.hh"

#include "G4Electron.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4VProcess.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr double kMeV = CLHEP::MeV;
constexpr double kCm = CLHEP::cm;

// Bin index with underflow at 0 and overflow at n + 1.
int EBin(const BinSpec& b, double e)
{
  return b.Index(e) + 1;
}

int CosBin(const BinSpec& b, double c)
{
  return std::clamp(b.Index(c), 0, b.n - 1);
}

// Track-length estimator for a particle losing energy linearly from eHi to eLo
// over a step: the step's weighted length wL is spread over the energy bins of
// row (n + 2 entries, with under/overflow) in proportion to the energy interval
// each bin covers. Energies in internal units.
void AddLinear(double* row, const BinSpec& b, double eHi, double eLo, double wL)
{
  if (eHi - eLo <= 1e-9 * kMeV) {
    row[EBin(b, eHi)] += wL;
    return;
  }
  const int i0 = EBin(b, eLo), i1 = EBin(b, eHi);
  if (i0 == i1) {
    row[i0] += wL;
    return;
  }
  const double perE = wL / (eHi - eLo);
  for (int i = i0; i <= i1; ++i) {
    // Bin i covers [lo, hi); under/overflow bins extend to the step limits.
    const double lo = (i == 0) ? eLo : b.Lo(i - 1);
    const double hi = (i == b.n + 1) ? eHi : b.Lo(i);
    const double overlap = std::min(hi, eHi) - std::max(lo, eLo);
    if (overlap > 0.) row[i] += perE * overlap;
  }
}

// As AddLinear, but the path within the step is spread as 1/L(E): the share of
// the step in [lo, hi) is (R(hi) - R(lo)) / (R(eHi) - R(eLo)), R the restricted
// CSDA range.
void AddCsda(double* row, const BinSpec& b, double eHi, double eLo, double wL, const RangeTable& range)
{
  const double rHi = range.Eval(eHi), rLo = range.Eval(eLo);
  if (eHi - eLo <= 1e-9 * kMeV || rHi - rLo <= 0.) {
    row[EBin(b, eHi)] += wL;
    return;
  }
  const int i0 = EBin(b, eLo), i1 = EBin(b, eHi);
  if (i0 == i1) {
    row[i0] += wL;
    return;
  }
  const double perR = wL / (rHi - rLo);
  for (int i = i0; i <= i1; ++i) {
    const double lo = std::max((i == 0) ? eLo : b.Lo(i - 1), eLo);
    const double hi = std::min((i == b.n + 1) ? eHi : b.Lo(i), eHi);
    if (hi > lo) row[i] += perR * (range.Eval(hi) - range.Eval(lo));
  }
}

int ParticleIndex(const G4ParticleDefinition* p)
{
  if (p == G4Gamma::Definition()) return 0;
  if (p == G4Electron::Definition()) return 1;
  if (p == G4Positron::Definition()) return 2;
  return -1;
}
}  // namespace

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  const Config& c = Config::Get();
  if (c.tableMode) return;
  Tallies& T = *fRun->GetTallies();

  G4Track* track = step->GetTrack();
  auto* info = static_cast<TrackInfo*>(track->GetUserInformation());
  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  const G4VPhysicalVolume* preVol = pre->GetPhysicalVolume();
  const bool inScoring = (preVol == DetectorConstruction::ScoringPV());
  const bool inPhantom = preVol != DetectorConstruction::WorldPV();  // phantom, importance cells, scoring
  const double w = pre->GetWeight();
  const G4ParticleDefinition* pd = track->GetDefinition();
  const int pIdx = ParticleIndex(pd);
  const G4VProcess* proc = post->GetProcessDefinedStep();
  // Physics interactions only: steps limited by transportation or by the
  // importance-sampling process at a parallel-world boundary are not interactions.
  const bool interaction = proc != nullptr && proc->GetProcessType() == fElectromagnetic;
  const G4String& procName = proc ? proc->GetProcessName() : G4String("none");
  const double stepLen = step->GetStepLength();
  const double edep = step->GetTotalEnergyDeposit();
  const double ePre = pre->GetKineticEnergy();
  const double ePost = post->GetKineticEnergy();
  const G4ThreeVector dirPre = pre->GetMomentumDirection();
  const auto* secs = step->GetSecondaryInCurrentStep();

  // Radiative energy lost in this step by a charged particle: bremsstrahlung
  // photons, and the kinetic energy carried off by in-flight annihilation.
  double eRad = 0.;
  if (pIdx == 1 || pIdx == 2) {
    double annihPhotons = 0.;
    for (const G4Track* sec : *secs) {
      const G4VProcess* cp = sec->GetCreatorProcess();
      if (cp == nullptr || sec->GetDefinition() != G4Gamma::Definition()) continue;
      if (cp->GetProcessName() == "eBrem") eRad += sec->GetKineticEnergy();
      else if (cp->GetProcessName() == "annihil") annihPhotons += sec->GetKineticEnergy();
    }
    if (annihPhotons > 0.) eRad += std::max(0., annihPhotons - 2. * CLHEP::electron_mass_c2);
    if (info != nullptr && info->kcolTag) T.current[2] += w * eRad / kMeV;
  }

  // ---- Scoring volume -----------------------------------------------------
  if (inScoring && info != nullptr) {
    if (pIdx == 0) {
      const int cat = info->photonCat;
      // Layer 1: photon fluence, energy fluence, angular distribution.
      const double wL = w * stepLen / kCm;
      T["photon_fluence"].at({cat, EBin(c.photonE, ePre)}) += wL;
      T["photon_energy_fluence"].at({cat}) += wL * ePre / kMeV;
      T["photon_track_length"].at({cat}) += wL;
      T.current[3] += wL;
      T["photon_angular"].at({cat, CosBin(c.cosTheta, dirPre.z())}) += wL;

      if (interaction) {
        // Layers 2 and 3, kerma, event sample.
        const int pi = ProcessIndexOf(procName);
        const bool alive = track->GetTrackStatus() == fAlive || track->GetTrackStatus() == fStopButAlive;
        double tCharged = 0., eSecPhot = 0., tMax = -1., cosE = std::numeric_limits<double>::quiet_NaN();
        double ePos = -1.;
        int nCharged = 0, pdgMax = 0, nPositrons = 0;
        for (const G4Track* sec : *secs) {
          const int si = ParticleIndex(sec->GetDefinition());
          const double ts = sec->GetKineticEnergy();
          if (si == 0) {
            eSecPhot += ts;
            continue;
          }
          if (si < 0) continue;
          ++nCharged;
          tCharged += ts;
          const int part = si - 1;  // 0: e-, 1: e+
          if (part == 1) {
            ++nPositrons;
            ePos = ts / kMeV;
          }
          T["initial_spectrum"].at({pi, cat, part, EBin(c.electronE, ts)}) += w;
          T["initial_count"].at({pi, cat, part}) += w;
          T["initial_energy"].at({pi, cat, part}) += w * ts / kMeV;
          T["initial_cos"].at({pi, IsPrimaryCat(cat) ? 0 : 1, CosBin(c.cosTheta, sec->GetMomentumDirection().z())}) += w;
          if (ts > tMax) {
            tMax = ts;
            pdgMax = sec->GetDefinition()->GetPDGEncoding();
            cosE = dirPre.dot(sec->GetMomentumDirection());
          }
        }
        const double eOut = alive ? ePost : 0.;
        const double eTr = tCharged + edep;
        T["interactions"].at({cat, pi}) += w;
        T["interaction_energy_transfer"].at({cat, pi}) += w * eTr / kMeV;
        T["interaction_incident_energy"].at({cat, pi}) += w * ePre / kMeV;
        T["interaction_spectrum"].at({pi, EBin(c.photonE, ePre)}) += w;
        T.current[1] += w * eTr / kMeV;
        T.s["interactions_unweighted"] += 1.;
        const double balance = ePre - eOut - eSecPhot - tCharged - edep - nPositrons * 2. * CLHEP::electron_mass_c2;
        T.energyBalanceMaxAbs = std::max(T.energyBalanceMaxAbs, std::abs(balance) / kMeV);

        if (static_cast<int>(T.events.size()) < T.eventSampleMax) {
          const G4ThreeVector x = post->GetPosition();
          EventRecord r;
          r.eventId = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
          r.weight = w;
          r.photonCat = cat;
          r.process = pi;
          r.eIn = ePre / kMeV;
          r.cosIn = dirPre.z();
          r.eOut = alive ? ePost / kMeV : -1.;
          r.cosScatter = alive ? dirPre.dot(post->GetMomentumDirection()) : std::numeric_limits<double>::quiet_NaN();
          r.nCharged = nCharged;
          r.tMax = tMax > 0. ? tMax / kMeV : 0.;
          r.pdgMax = pdgMax;
          r.cosElectron = cosE;
          r.tChargedSum = tCharged / kMeV;
          r.ePositron = ePos;
          r.eSecPhotons = eSecPhot / kMeV;
          r.eLocal = edep / kMeV;
          r.x = x.x() / kCm;
          r.y = x.y() / kCm;
          r.z = x.z() / kCm;
          T.events.push_back(r);
        }
      }
    }
    else if (pIdx == 1 || pIdx == 2) {
      // Layer 4: track length spread over the energy interval of the
      // continuous loss in this step (E_pre to E_pre - edep, bounded by E_post).
      const double eLo = std::clamp(ePre - edep, ePost, ePre);
      const double wL = w * stepLen / kCm;
      if (pIdx == 1) {
        AddLinear(T["electron_fluence"].data.data(), c.electronE, ePre, eLo, wL);
        if (info->delta) AddLinear(T["electron_fluence_delta"].data.data(), c.electronE, ePre, eLo, wL);
        const int byParent = info->parentCat < 0 ? 2 : (IsPrimaryCat(info->parentCat) ? 0 : 1);
        AddLinear(&T["electron_fluence_by_parent"].at({byParent, 0}), c.electronE, ePre, eLo, wL);
        const RangeTable& rt = fRun->Range(0);
        AddCsda(T["electron_fluence_csda"].data.data(), c.electronE, ePre, eLo, wL, rt);
        if (info->delta) AddCsda(T["electron_fluence_delta_csda"].data.data(), c.electronE, ePre, eLo, wL, rt);
        AddCsda(&T["electron_fluence_by_parent_csda"].at({byParent, 0}), c.electronE, ePre, eLo, wL, rt);
        T["electron_angular"].at({CosBin(c.cosTheta, dirPre.z())}) += wL;
      }
      else {
        AddLinear(T["positron_fluence"].data.data(), c.electronE, ePre, eLo, wL);
        AddCsda(T["positron_fluence_csda"].data.data(), c.electronE, ePre, eLo, wL, fRun->Range(1));
        T["positron_angular"].at({CosBin(c.cosTheta, dirPre.z())}) += wL;
      }
      const double eLost = ePre - ePost;
      T["charged_energy_loss"].at({pIdx - 1, 0}) += w * (eLost - eRad) / kMeV;
      T["charged_energy_loss"].at({pIdx - 1, 1}) += w * eRad / kMeV;
    }

    // Layer 5: energy deposited, by parent photon category and depositing particle.
    if (edep > 0. && pIdx >= 0) {
      const int parent = (pIdx == 0) ? info->photonCat : (info->parentCat < 0 ? kNPhotonCat : info->parentCat);
      T["edep_by_parent"].at({parent, pIdx}) += w * edep / kMeV;
      T.current[0] += w * edep / kMeV;
    }
  }

  // ---- Central-axis depth scan (anywhere in the phantom) -------------------
  if (c.depthScan && inPhantom && edep > 0.) {
    const G4ThreeVector p = pre->GetPosition() + G4UniformRand() * (post->GetPosition() - pre->GetPosition());
    if (std::hypot(p.x(), p.y()) < c.depthRadius) {
      T["depth_edep"].data[EBin(c.depth, p.z())] += w * edep / kMeV;
    }
  }

  // ---- Track information for secondaries created in this step -------------
  if (info != nullptr) {
    const bool photonInteractionInScoring = (pIdx == 0) && interaction && inScoring;
    for (const G4Track* sec : *secs) {
      auto* si = new TrackInfo();
      const bool secIsPhoton = sec->GetDefinition() == G4Gamma::Definition();
      const G4VProcess* creator = sec->GetCreatorProcess();
      const bool splitCopy = creator != nullptr && creator->GetProcessType() != fElectromagnetic;
      if (splitCopy) {
        // Copy made by importance splitting: the same photon, same history.
        *si = *info;
      }
      else if (secIsPhoton) {
        si->photonCat = kSecondaryPhoton;
      }
      else if (pIdx == 0) {
        si->parentCat = info->photonCat;
        si->kcolTag = photonInteractionInScoring;
      }
      else {
        si->parentCat = info->parentCat;
        si->delta = true;
        si->kcolTag = info->kcolTag;
      }
      const_cast<G4Track*>(sec)->SetUserInformation(si);
    }
    // A source photon that interacted is no longer primary.
    if (pIdx == 0 && interaction && IsPrimaryCat(info->photonCat)) info->photonCat = kScatteredSource;
  }
}
