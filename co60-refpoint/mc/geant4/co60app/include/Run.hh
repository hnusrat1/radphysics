#ifndef CO60_RUN_HH
#define CO60_RUN_HH

#include "Tallies.hh"

#include "G4Run.hh"

// Per-thread tallies; worker runs are merged into the master run.
class Run : public G4Run
{
 public:
  Run();
  void Merge(const G4Run* other) override;
  Tallies tallies;
};

#endif
