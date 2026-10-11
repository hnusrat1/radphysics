#include "PrimaryGeneratorAction.hh"

#include "Config.hh"

#include "G4Event.hh"
#include "G4Gamma.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <cmath>

PrimaryGeneratorAction::PrimaryGeneratorAction() : fGun(new G4ParticleGun(1))
{
  fGun->SetParticleDefinition(G4Gamma::Definition());
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fGun;
}

G4double PrimaryGeneratorAction::PyramidSolidAngle()
{
  const Config& c = Config::Get();
  if (c.fieldRadius > 0.) {
    // Circular field (B0): cone of half-angle atan(r / SSD).
    return CLHEP::twopi * (1. - c.ssd / std::sqrt(c.ssd * c.ssd + c.fieldRadius * c.fieldRadius));
  }
  const G4double tx = 0.5 * c.fieldX / c.ssd;
  const G4double ty = 0.5 * c.fieldY / c.ssd;
  return 4. * std::asin(tx * ty / std::sqrt((1. + tx * tx) * (1. + ty * ty)));
}

G4double PrimaryGeneratorAction::FluenceOnAxis(G4double depth)
{
  const G4double r = Config::Get().ssd + depth;
  return 1. / (PyramidSolidAngle() * r * r);
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  const Config& c = Config::Get();
  if (c.tableMode) {
    fGun->SetParticleEnergy(1. * MeV);
    fGun->SetParticlePosition({0., 0., -45. * cm});
    fGun->SetParticleMomentumDirection({0., 0., 1.});
    fGun->GeneratePrimaryVertex(event);
    return;
  }

  if (c.fieldRadius > 0.) {
    // Circular field (B0): uniform in solid angle inside the cone.
    const G4double cosMin = c.ssd / std::sqrt(c.ssd * c.ssd + c.fieldRadius * c.fieldRadius);
    const G4double dz = cosMin + (1. - cosMin) * G4UniformRand();
    const G4double sinT = std::sqrt((1. - dz) * (1. + dz));
    const G4double phi = CLHEP::twopi * G4UniformRand();
    fGun->SetParticleEnergy(G4UniformRand() < c.line1Probability ? c.line1 : c.line2);
    fGun->SetParticlePosition({0., 0., -c.ssd});
    fGun->SetParticleMomentumDirection({sinT * std::cos(phi), sinT * std::sin(phi), dz});
    fGun->GeneratePrimaryVertex(event);
    return;
  }

  // Uniform in solid angle on the cap that contains the pyramid, keeping
  // directions inside the pyramid (rejection).
  const G4double tx = 0.5 * c.fieldX / c.ssd;
  const G4double ty = 0.5 * c.fieldY / c.ssd;
  const G4double cosMin = 1. / std::sqrt(1. + tx * tx + ty * ty);
  G4double dx, dy, dz;
  do {
    dz = cosMin + (1. - cosMin) * G4UniformRand();
    const G4double sinT = std::sqrt((1. - dz) * (1. + dz));
    const G4double phi = CLHEP::twopi * G4UniformRand();
    dx = sinT * std::cos(phi);
    dy = sinT * std::sin(phi);
  } while (std::abs(dx) > tx * dz || std::abs(dy) > ty * dz);

  fGun->SetParticleEnergy(G4UniformRand() < c.line1Probability ? c.line1 : c.line2);
  fGun->SetParticlePosition({0., 0., -c.ssd});
  fGun->SetParticleMomentumDirection({dx, dy, dz});
  fGun->GeneratePrimaryVertex(event);
}
