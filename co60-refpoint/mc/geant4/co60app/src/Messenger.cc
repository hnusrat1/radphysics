#include "Messenger.hh"

#include "Config.hh"
#include "ImportanceWorld.hh"

#include "G4Exception.hh"
#include "G4UIcommand.hh"
#include "G4UIdirectory.hh"
#include "G4UIparameter.hh"
#include "G4UnitsTable.hh"

#include <sstream>
#include <vector>

namespace
{
std::vector<G4String> Tokens(const G4String& s)
{
  std::istringstream is(s);
  std::vector<G4String> out;
  G4String t;
  while (is >> t) out.push_back(t);
  return out;
}

G4double WithUnit(const G4String& value, const G4String& unit)
{
  return std::stod(value) * G4UnitDefinition::GetValueOf(unit);
}

G4bool ToBool(const G4String& s)
{
  return G4UIcommand::ConvertToBool(s);
}

void Fail(const G4String& cmd, const G4String& msg)
{
  G4Exception("Messenger", "CO60-UI", FatalErrorInArgument, (cmd + ": " + msg).c_str());
}
}  // namespace

Messenger::Messenger()
{
  const char* dirs[] = {"/co60/", "/co60/geom/", "/co60/source/", "/co60/phys/",
                        "/co60/vr/", "/co60/bins/", "/co60/out/"};
  for (int i = 0; i < 7; ++i) {
    fDirs[i] = new G4UIdirectory(dirs[i], false);
  }
  Make("/co60/geom/phantomSize", "Phantom cube edge", "value unit");
  Make("/co60/geom/scoringRadius", "Scoring cylinder radius", "value unit");
  Make("/co60/geom/scoringThickness", "Scoring cylinder thickness", "value unit");
  Make("/co60/geom/scoringDepth", "Depth of the scoring cylinder centre", "value unit");
  Make("/co60/geom/tableMode", "Geometry for stopping-power tables only", "bool");
  Make("/co60/source/ssd", "Source to surface distance", "value unit");
  Make("/co60/source/field", "Field size at the phantom surface", "x y unit");
  Make("/co60/source/lines", "Two photon lines and the probability of the first", "e1 e2 unit p1");
  Make("/co60/phys/electronThreshold", "e-/e+ production threshold in G4_WATER", "value unit");
  Make("/co60/phys/gammaThreshold", "Upper bound on the gamma production threshold in G4_WATER", "value unit");
  Make("/co60/vr/russianRoulette", "Russian roulette of electrons born outside the ROI", "bool");
  Make("/co60/vr/survival", "Survival probability for Russian roulette", "p");
  Make("/co60/vr/roi", "ROI cylinder on the beam axis", "radius zmin zmax unit");
  Make("/co60/vr/importanceCell", "Append a nested importance cell (outermost first)", "radius zmin zmax unit importance");
  Make("/co60/vr/applyImportance", "Fill the importance store (after /run/initialize)", "none");
  Make("/co60/bins/photonE", "Photon energy grid", "min max n unit");
  Make("/co60/bins/electronE", "Electron energy grid", "min max n unit");
  Make("/co60/bins/cosTheta", "Number of cos(theta) bins on [-1, 1]", "n");
  Make("/co60/bins/depth", "Depth-scan slabs and radius", "min max n radius unit");
  Make("/co60/score/depthScan", "Score the central-axis depth scan", "bool");
  Make("/co60/out/file", "Output JSON file", "path");
  Make("/co60/out/eventSample", "Event-sample CSV file and maximum number of interactions", "path max");
}

Messenger::~Messenger()
{
  for (auto& c : fCommands) delete c.second;
  for (auto* d : fDirs) delete d;
}

G4UIcommand* Messenger::Make(const G4String& path, const G4String& guidance, const G4String& params)
{
  auto* cmd = new G4UIcommand(path.c_str(), this);
  cmd->SetGuidance((guidance + " [" + params + "]").c_str());
  auto* p = new G4UIparameter("args", 's', params == "none");
  if (params == "none") p->SetDefaultValue("");
  cmd->SetParameter(p);
  cmd->SetToBeBroadcasted(false);
  fCommands[path] = cmd;
  return cmd;
}

void Messenger::SetNewValue(G4UIcommand* command, G4String value)
{
  // Each command has one string parameter; G4UIcommand passes it the rest of
  // the line, which is split into tokens here.
  Config& c = Config::Get();
  const G4String path = command->GetCommandPath();
  auto t = Tokens(value);
  auto need = [&](size_t n) {
    if (t.size() != n) Fail(path, "expected " + std::to_string(n) + " arguments, got '" + value + "'");
  };

  if (path == "/co60/geom/phantomSize") { need(2); c.phantomSize = WithUnit(t[0], t[1]); }
  else if (path == "/co60/geom/scoringRadius") { need(2); c.scoringRadius = WithUnit(t[0], t[1]); }
  else if (path == "/co60/geom/scoringThickness") { need(2); c.scoringThickness = WithUnit(t[0], t[1]); }
  else if (path == "/co60/geom/scoringDepth") { need(2); c.scoringDepth = WithUnit(t[0], t[1]); }
  else if (path == "/co60/geom/tableMode") { need(1); c.tableMode = ToBool(t[0]); }
  else if (path == "/co60/source/ssd") { need(2); c.ssd = WithUnit(t[0], t[1]); }
  else if (path == "/co60/source/field") {
    need(3);
    c.fieldX = WithUnit(t[0], t[2]);
    c.fieldY = WithUnit(t[1], t[2]);
  }
  else if (path == "/co60/source/lines") {
    need(4);
    c.line1 = WithUnit(t[0], t[2]);
    c.line2 = WithUnit(t[1], t[2]);
    c.line1Probability = std::stod(t[3]);
  }
  else if (path == "/co60/phys/electronThreshold") { need(2); c.electronThreshold = WithUnit(t[0], t[1]); }
  else if (path == "/co60/phys/gammaThreshold") { need(2); c.gammaThreshold = WithUnit(t[0], t[1]); }
  else if (path == "/co60/vr/russianRoulette") { need(1); c.rrEnabled = ToBool(t[0]); }
  else if (path == "/co60/vr/survival") {
    need(1);
    c.rrSurvival = std::stod(t[0]);
    if (c.rrSurvival <= 0. || c.rrSurvival > 1.) Fail(path, "survival probability must be in (0, 1]");
  }
  else if (path == "/co60/vr/roi") {
    need(4);
    c.roiRadius = WithUnit(t[0], t[3]);
    c.roiZmin = WithUnit(t[1], t[3]);
    c.roiZmax = WithUnit(t[2], t[3]);
  }
  else if (path == "/co60/vr/importanceCell") {
    need(5);
    c.importanceCells.push_back(
      {WithUnit(t[0], t[3]), WithUnit(t[1], t[3]), WithUnit(t[2], t[3]), std::stod(t[4])});
  }
  else if (path == "/co60/vr/applyImportance") {
    if (!c.importanceEnabled || ImportanceWorld::Instance() == nullptr) {
      Fail(path, "importance sampling was not enabled at launch (--importance)");
    }
    ImportanceWorld::Instance()->CreateImportanceStore();
  }
  else if (path == "/co60/bins/photonE" || path == "/co60/bins/electronE") {
    need(4);
    BinSpec b{WithUnit(t[0], t[3]), WithUnit(t[1], t[3]), std::stoi(t[2])};
    (path == "/co60/bins/photonE" ? c.photonE : c.electronE) = b;
  }
  else if (path == "/co60/bins/cosTheta") { need(1); c.cosTheta = BinSpec{-1., 1., std::stoi(t[0])}; }
  else if (path == "/co60/bins/depth") {
    need(5);
    c.depth = BinSpec{WithUnit(t[0], t[4]), WithUnit(t[1], t[4]), std::stoi(t[2])};
    c.depthRadius = WithUnit(t[3], t[4]);
  }
  else if (path == "/co60/score/depthScan") { need(1); c.depthScan = ToBool(t[0]); }
  else if (path == "/co60/out/file") { need(1); c.outputFile = t[0]; }
  else if (path == "/co60/out/eventSample") {
    need(2);
    c.eventSampleFile = t[0];
    c.eventSampleMax = std::stoi(t[1]);
  }
  else {
    Fail(path, "unknown command");
  }
}
