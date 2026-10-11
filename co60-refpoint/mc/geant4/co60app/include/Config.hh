// Run configuration shared by all threads.
//
// Values are set by UI commands on the master thread before /run/initialize
// (geometry, cuts) or before /run/beamOn (everything else). Worker threads only
// read them. Commands are not broadcast to workers, so there is one writer.

#ifndef CO60_CONFIG_HH
#define CO60_CONFIG_HH

#include "G4String.hh"
#include "G4SystemOfUnits.hh"

#include <string>
#include <vector>

struct BinSpec
{
  G4double min = 0.;
  G4double max = 1.;
  G4int n = 1;

  G4int Index(G4double x) const
  {
    // -1: underflow, n: overflow
    if (x < min) return -1;
    if (x >= max) return n;
    return static_cast<G4int>((x - min) / (max - min) * n);
  }
  G4double Width() const { return (max - min) / n; }
  G4double Lo(G4int i) const { return min + i * Width(); }
};

struct ImportanceCell
{
  G4double radius;
  G4double zmin;
  G4double zmax;
  G4double importance;
};

struct Config
{
  static Config& Get()
  {
    static Config instance;
    return instance;
  }

  // Geometry (R0 defaults, STUDY_PLAN Section 2). Depth is measured from the phantom surface at z = 0.
  G4double phantomSize = 30. * cm;      // cube edge
  G4double scoringRadius = 1.0 * cm;
  G4double scoringThickness = 0.2 * cm;
  G4double scoringDepth = 5.0 * cm;     // centre of the scoring cylinder
  G4bool tableMode = false;             // geometry with water and air slabs for stopping-power tables
  // Benchmark geometry B0 (STUDY_PLAN Section 4.2): cylindrical phantom on the beam axis instead of the cube.
  G4bool cylinderPhantom = false;
  G4double phantomRadius = 15. * cm;
  G4double phantomDepth = 30. * cm;
  G4double PhantomDepth() const { return cylinderPhantom ? phantomDepth : phantomSize; }
  G4double PhantomHalfXY() const { return cylinderPhantom ? phantomRadius : 0.5 * phantomSize; }

  // Source model A
  G4double ssd = 100. * cm;
  G4double fieldX = 10. * cm;           // at the phantom surface
  G4double fieldY = 10. * cm;
  G4double fieldRadius = 0.;            // > 0: circular field of this radius at the surface (B0), not the square
  G4double line1 = 1.1732 * MeV;
  G4double line2 = 1.3325 * MeV;
  G4double line1Probability = 0.5;

  // Production thresholds in G4_WATER, converted to range cuts at initialization
  G4double electronThreshold = 10. * keV;
  G4double gammaThreshold = 10. * keV;  // upper bound; the range cut for gamma is the electron one if lower

  // Variance reduction: Russian roulette of electrons born outside the region of interest (ROI).
  G4bool rrEnabled = false;
  G4double rrSurvival = 1. / 20.;
  G4double roiRadius = 1.8 * cm;
  G4double roiZmin = 4.1 * cm;
  G4double roiZmax = 5.9 * cm;

  // Photon importance sampling (enabled by the --importance or --importance-mass
  // launch flag): nested cylinders, outermost first; the rest of the world has
  // importance 1. With --importance the cells are a parallel world; with
  // --importance-mass they are water volumes in the mass geometry (DECISIONS D021).
  G4bool importanceEnabled = false;
  G4bool importanceMassWorld = false;
  std::vector<ImportanceCell> importanceCells;

  // Scoring grids (from config/bins.yaml via the run harness)
  BinSpec photonE{0., 1.4 * MeV, 140};
  BinSpec electronE{0., 1.4 * MeV, 1400};
  BinSpec cosTheta{-1., 1., 40};
  BinSpec depth{0., 10. * cm, 100};
  G4double depthRadius = 1.0 * cm;
  G4bool depthScan = false;

  // Output
  G4String outputFile = "co60app_output.json";
  G4String eventSampleFile = "";        // empty: no event sample
  G4int eventSampleMax = 0;

  G4bool InRoi(G4double r, G4double z) const
  {
    return r <= roiRadius && z >= roiZmin && z <= roiZmax;
  }

 private:
  Config() = default;
};

#endif
