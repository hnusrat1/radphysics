#ifndef CO60_TRACKINFO_HH
#define CO60_TRACKINFO_HH

#include "G4VUserTrackInformation.hh"

// Photon categories (STUDY_PLAN Layer 1 splits primary and scattered; the
// non-primary part is further split into scattered source photons and photons
// created by other particles).
enum PhotonCat : int
{
  kPrimaryLine1 = 0,    // source photon, 1.1732 MeV, no interaction since leaving the source
  kPrimaryLine2 = 1,    // source photon, 1.3325 MeV, no interaction since leaving the source
  kScatteredSource = 2, // source photon after one or more interactions (Compton or Rayleigh)
  kSecondaryPhoton = 3, // bremsstrahlung, fluorescence or annihilation photon
  kNPhotonCat = 4
};

inline bool IsPrimaryCat(int cat) { return cat == kPrimaryLine1 || cat == kPrimaryLine2; }

class TrackInfo : public G4VUserTrackInformation
{
 public:
  // Photons: current category, updated after each interaction.
  int photonCat = kSecondaryPhoton;
  // Charged particles: category of the photon whose interaction set the chain
  // in motion (-1 if unknown), whether the particle was created by another
  // charged particle (delta ray or descendant), and whether the chain started
  // with a photon interaction inside the scoring volume (used for collision kerma).
  int parentCat = -1;
  bool delta = false;
  bool kcolTag = false;
};

#endif
