#include "Actions.hh"
#include "Config.hh"
#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "Run.hh"

#include "G4Electron.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4Exception.hh"
#include "G4Gamma.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4NistManager.hh"
#include "G4ProductionCutsTable.hh"
#include "G4RunManagerKernel.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

G4String RunAction::gPhysicsName = "option4";

namespace
{
// CSDA range of an electron in G4_WATER from the production physics:
// R(E) = integral of dE / S_total(E) from 1 keV, trapezoidal on a log grid.
G4double CsdaRangeWater(G4double e)
{
  G4EmCalculator calc;
  const G4Material* water = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  const int n = 400;
  const G4double lo = std::log(1. * keV), hi = std::log(e);
  G4double r = 0., prevE = 1. * keV;
  G4double prevInv = 1. / calc.ComputeTotalDEDX(prevE, G4Electron::Definition(), water);
  for (int i = 1; i <= n; ++i) {
    const G4double ei = std::exp(lo + (hi - lo) * i / n);
    const G4double inv = 1. / calc.ComputeTotalDEDX(ei, G4Electron::Definition(), water);
    r += 0.5 * (inv + prevInv) * (ei - prevE);
    prevE = ei;
    prevInv = inv;
  }
  return r;
}

const G4MaterialCutsCouple* WaterCouple()
{
  auto* table = G4ProductionCutsTable::GetProductionCutsTable();
  for (size_t i = 0; i < table->GetTableSize(); ++i) {
    const auto* couple = table->GetMaterialCutsCouple(static_cast<G4int>(i));
    if (couple->GetMaterial()->GetName() == "G4_WATER") return couple;
  }
  return nullptr;
}

void WriteBins(std::ostream& os, const char* name, const BinSpec& b, double unit, const char* unitName, bool last)
{
  os << "    \"" << name << "\": {\"min\": " << b.min / unit << ", \"max\": " << b.max / unit << ", \"n\": " << b.n
     << ", \"unit\": \"" << unitName << "\"}" << (last ? "\n" : ",\n");
}
}  // namespace

G4Run* RunAction::GenerateRun()
{
  auto* run = new Run();
  fTallies = &run->tallies;
  return run;
}

void RunAction::CheckRoi() const
{
  const Config& c = Config::Get();
  if (!c.rrEnabled) return;
  // Electrons born outside the ROI must not be able to reach the scoring
  // volume (or the depth-scan column). The required buffer is 1.2 times the
  // CSDA range in water of an electron carrying the full energy of the higher
  // line; the factor covers range straggling from energy-loss fluctuations.
  const G4double csda = CsdaRangeWater(std::max(c.line1, c.line2));
  const G4double buffer = 1.2 * csda;
  const G4double zLo = c.scoringDepth - 0.5 * c.scoringThickness, zHi = c.scoringDepth + 0.5 * c.scoringThickness;
  std::ostringstream msg;
  msg << "ROI must extend at least 1.2 x CSDA range (" << buffer / mm << " mm) beyond every scored region";
  G4bool ok = c.roiRadius >= c.scoringRadius + buffer && c.roiZmin <= zLo - buffer && c.roiZmax >= zHi + buffer;
  if (c.depthScan) {
    ok = ok && c.roiRadius >= c.depthRadius + buffer && c.roiZmin <= c.depth.min && c.roiZmax >= c.depth.max + buffer;
  }
  if (!ok) G4Exception("RunAction", "CO60-ROI", FatalException, msg.str().c_str());
  G4cout << "ROI check passed: buffer needed " << buffer / mm << " mm (1.2 x CSDA range " << csda / mm << " mm of "
         << std::max(c.line1, c.line2) / MeV << " MeV e- in G4_WATER)" << G4endl;
}

void RunAction::BeginOfRunAction(const G4Run*)
{
  if (!IsMaster()) return;
  fTimer.Start();
  if (!Config::Get().tableMode) CheckRoi();
}

void RunAction::EndOfRunAction(const G4Run* run)
{
  if (!IsMaster()) return;
  fTimer.Stop();
  if (Config::Get().tableMode) {
    WriteTables();
    return;
  }
  const Tallies& t = static_cast<const Run*>(run)->tallies;
  WriteOutput(run, t);
  if (!Config::Get().eventSampleFile.empty()) WriteEventSample(t);
}

void RunAction::WriteOutput(const G4Run* run, const Tallies& t) const
{
  const Config& c = Config::Get();
  const G4Material* water = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  const auto* couple = WaterCouple();
  auto* cuts = G4ProductionCutsTable::GetProductionCutsTable();
  const auto* params = G4EmParameters::Instance();
  const G4double volume = CLHEP::pi * c.scoringRadius * c.scoringRadius * c.scoringThickness;

  const G4String tmp = c.outputFile + ".tmp";
  std::ofstream os(tmp);
  os << std::setprecision(std::numeric_limits<double>::max_digits10);
  os << "{\n  \"format\": \"co60app-1\",\n";
  os << "  \"units\": {\"length\": \"cm\", \"energy\": \"MeV\", \"mass\": \"g\", \"fluence\": \"cm^-2\"},\n";
  os << "  \"geant4_version\": \"" << G4RunManagerKernel::GetRunManagerKernel()->GetVersionString() << "\",\n";
  os << "  \"physics\": \"" << gPhysicsName << "\",\n";
  os << "  \"histories\": " << run->GetNumberOfEvent() << ",\n";
  os << "  \"wall_time_s\": " << fTimer.GetRealElapsed() << ",\n";
  os << "  \"threads\": " << G4Threading::GetNumberOfRunningWorkerThreads() << ",\n";
  os << "  \"geometry\": {\"phantom_size\": " << c.phantomSize / cm << ", \"scoring_radius\": " << c.scoringRadius / cm
     << ", \"scoring_thickness\": " << c.scoringThickness / cm << ", \"scoring_depth\": " << c.scoringDepth / cm
     << ", \"scoring_volume_cm3\": " << volume / cm3
     << ", \"scoring_mass_g\": " << DetectorConstruction::ScoringMass() / g
     << ", \"water_density_g_cm3\": " << water->GetDensity() / (g / cm3)
     << ", \"water_imean_eV\": " << water->GetIonisation()->GetMeanExcitationEnergy() / eV << "},\n";
  os << "  \"source\": {\"ssd\": " << c.ssd / cm << ", \"field_x\": " << c.fieldX / cm << ", \"field_y\": "
     << c.fieldY / cm << ", \"line1_MeV\": " << c.line1 / MeV << ", \"line2_MeV\": " << c.line2 / MeV
     << ", \"line1_probability\": " << c.line1Probability
     << ", \"solid_angle_sr\": " << PrimaryGeneratorAction::PyramidSolidAngle()
     << ", \"phi0_per_history_cm-2\": " << PrimaryGeneratorAction::FluenceOnAxis(c.scoringDepth) * cm2 << "},\n";
  os << "  \"cuts_g4_water\": {";
  if (couple != nullptr) {
    const int idx = couple->GetIndex();
    const char* names[3] = {"gamma", "e-", "e+"};
    for (int p = 0; p < 3; ++p) {
      os << "\"" << names[p] << "_range_cm\": " << couple->GetProductionCuts()->GetProductionCut(p) / cm << ", \""
         << names[p] << "_threshold_MeV\": " << (*cuts->GetEnergyCutsVector(p))[idx] / MeV << (p < 2 ? ", " : "");
    }
  }
  os << "},\n";
  os << "  \"em_parameters\": {\"msc_range_factor\": " << params->MscRangeFactor() << ", \"msc_skin\": "
     << params->MscSkin() << ", \"msc_step_limit_type\": " << params->MscStepLimitType()
     << ", \"lowest_electron_energy_MeV\": " << params->LowestElectronEnergy() / MeV
     << ", \"mott_correction\": " << (params->UseMottCorrection() ? "true" : "false")
     << ", \"fluorescence\": " << (params->Fluo() ? "true" : "false")
     << ", \"auger\": " << (params->Auger() ? "true" : "false") << "},\n";
  os << "  \"variance_reduction\": {\"russian_roulette\": " << (c.rrEnabled ? "true" : "false")
     << ", \"survival\": " << c.rrSurvival << ", \"roi_radius\": " << c.roiRadius / cm << ", \"roi_zmin\": "
     << c.roiZmin / cm << ", \"roi_zmax\": " << c.roiZmax / cm
     << ", \"photon_importance_sampling\": " << (c.importanceEnabled ? "true" : "false") << ", \"importance_cells\": [";
  for (size_t i = 0; i < c.importanceCells.size(); ++i) {
    const auto& ic = c.importanceCells[i];
    os << (i ? ", " : "") << "{\"radius\": " << ic.radius / cm << ", \"zmin\": " << ic.zmin / cm
       << ", \"zmax\": " << ic.zmax / cm << ", \"importance\": " << ic.importance << "}";
  }
  os << "]},\n";
  os << "  \"depth_scan\": {\"enabled\": " << (c.depthScan ? "true" : "false") << ", \"radius\": "
     << c.depthRadius / cm << "},\n";
  os << "  \"bins\": {\n";
  WriteBins(os, "photon_energy", c.photonE, MeV, "MeV", false);
  WriteBins(os, "electron_energy", c.electronE, MeV, "MeV", false);
  WriteBins(os, "cos_theta", c.cosTheta, 1., "1", false);
  WriteBins(os, "depth", c.depth, cm, "cm", true);
  os << "  },\n";
  os << "  \"energy_bins_note\": \"energy and depth tallies have n + 2 entries: underflow, n bins, overflow\",\n";
  os << "  \"photon_categories\": [\"primary_line1\", \"primary_line2\", \"scattered_source\", \"secondary_photon\"],\n";
  os << "  \"processes\": [\"compt\", \"phot\", \"conv\", \"Rayl\", \"other\"],\n";
  t.WriteJson(os);
  os << "}\n";
  os.close();
  std::rename(tmp.c_str(), c.outputFile.c_str());
  G4cout << "Wrote " << c.outputFile << G4endl;
}

void RunAction::WriteEventSample(const Tallies& t) const
{
  const Config& c = Config::Get();
  std::ofstream os(c.eventSampleFile);
  os << std::setprecision(10);
  os << "event_id,weight,photon_cat,process,e_in_MeV,cos_in,e_out_MeV,cos_scatter,n_charged,t_max_MeV,pdg_max,"
        "cos_electron,t_charged_sum_MeV,t_positron_MeV,e_fluorescence_MeV,e_local_MeV,x_cm,y_cm,z_cm\n";
  for (const auto& r : t.events) {
    os << r.eventId << ',' << r.weight << ',' << r.photonCat << ',' << kProcessNames[r.process] << ',' << r.eIn << ','
       << r.cosIn << ',' << r.eOut << ',' << r.cosScatter << ',' << r.nCharged << ',' << r.tMax << ',' << r.pdgMax
       << ',' << r.cosElectron << ',' << r.tChargedSum << ',' << r.ePositron << ',' << r.eSecPhotons << ','
       << r.eLocal << ',' << r.x << ',' << r.y << ',' << r.z << '\n';
  }
}

void RunAction::WriteTables() const
{
  // Stopping powers and photon cross sections from the production physics, for
  // the Spencer-Attix calculation and the attenuation checks.
  const Config& c = Config::Get();
  G4EmCalculator calc;
  auto* nist = G4NistManager::Instance();
  const G4Material* mats[2] = {nist->FindOrBuildMaterial("G4_WATER"), nist->FindOrBuildMaterial("G4_AIR")};
  const auto* e = G4Electron::Definition();
  const auto* gamma = G4Gamma::Definition();
  const int n = 400;
  const G4double lo = std::log(100. * eV), hi = std::log(2. * MeV);

  std::ofstream os(c.outputFile);
  os << std::setprecision(std::numeric_limits<double>::max_digits10);
  os << "{\n  \"format\": \"co60app-tables-1\",\n  \"physics\": \"" << gPhysicsName << "\",\n";
  os << "  \"geant4_version\": \"" << G4RunManagerKernel::GetRunManagerKernel()->GetVersionString() << "\",\n";
  os << "  \"units\": {\"energy\": \"MeV\", \"stopping_power\": \"MeV cm2/g\", \"mass_attenuation\": \"cm2/g\"},\n";
  os << "  \"materials\": {\n";
  for (int im = 0; im < 2; ++im) {
    const G4Material* mat = mats[im];
    const G4double rho = mat->GetDensity();
    const G4double massUnit = MeV * cm2 / g;
    os << "    \"" << mat->GetName() << "\": {\"density_g_cm3\": " << rho / (g / cm3)
       << ", \"imean_eV\": " << mat->GetIonisation()->GetMeanExcitationEnergy() / eV
       << ", \"electron_density_per_cm3\": " << mat->GetElectronDensity() * cm3 << ", \"elements\": [";
    for (size_t k = 0; k < mat->GetNumberOfElements(); ++k) {
      const G4Element* el = mat->GetElement(static_cast<G4int>(k));
      os << (k ? ", " : "") << "{\"Z\": " << el->GetZ() << ", \"A_g_mol\": " << el->GetA() / (g / mole)
         << ", \"mass_fraction\": " << mat->GetFractionVector()[k] << "}";
    }
    os << "],\n      \"electron\": [\n";
    for (int i = 0; i <= n; ++i) {
      const G4double en = std::exp(lo + (hi - lo) * i / n);
      const G4double sCol = calc.ComputeDEDX(en, e, "eIoni", mat, en) / rho / massUnit;
      const G4double l10 = calc.ComputeDEDX(en, e, "eIoni", mat, 10. * keV) / rho / massUnit;
      const G4double l5 = calc.ComputeDEDX(en, e, "eIoni", mat, 5. * keV) / rho / massUnit;
      const G4double sRad = calc.ComputeDEDX(en, e, "eBrem", mat, en) / rho / massUnit;
      // Density-effect correction used by the Bethe-type models, x = log10(beta gamma)
      const G4double tau = en / CLHEP::electron_mass_c2;
      const G4double x = std::log10(std::sqrt(tau * (tau + 2.)));
      const G4double delta = mat->GetIonisation()->DensityCorrection(x);
      os << "        {\"E\": " << en / MeV << ", \"S_col\": " << sCol << ", \"L_10keV\": " << l10
         << ", \"L_5keV\": " << l5 << ", \"S_rad\": " << sRad << ", \"delta\": " << delta << "}"
         << (i < n ? ",\n" : "\n");
    }
    // Density-effect correction on a wide grid (1 keV to 10 GeV) for building
    // matched EGSnrc density-correction files (DECISIONS D015).
    os << "      ],\n      \"density_effect\": [\n";
    for (int i = 0; i <= 200; ++i) {
      const G4double en = std::exp(std::log(1. * keV) + (std::log(10. * GeV) - std::log(1. * keV)) * i / 200.);
      const G4double tau = en / CLHEP::electron_mass_c2;
      const G4double x = std::log10(std::sqrt(tau * (tau + 2.)));
      os << "        {\"E\": " << en / MeV << ", \"delta\": " << mat->GetIonisation()->DensityCorrection(x) << "}"
         << (i < 200 ? ",\n" : "\n");
    }
    os << "      ],\n      \"photon\": [\n";
    std::vector<G4double> energies;
    for (int i = 0; i <= n; ++i) energies.push_back(std::exp(lo + (hi - lo) * i / n));
    energies.push_back(1.1732 * MeV);
    energies.push_back(1.3325 * MeV);
    for (size_t i = 0; i < energies.size(); ++i) {
      const G4double en = energies[i];
      os << "        {\"E\": " << en / MeV;
      for (const char* p : {"compt", "phot", "conv", "Rayl"}) {
        os << ", \"" << p << "\": " << calc.ComputeCrossSectionPerVolume(en, gamma, p, mat) / rho / (cm2 / g);
      }
      os << "}" << (i + 1 < energies.size() ? ",\n" : "\n");
    }
    os << "      ]\n    }" << (im == 0 ? ",\n" : "\n");
  }
  os << "  }\n}\n";
  G4cout << "Wrote tables to " << c.outputFile << G4endl;
}
