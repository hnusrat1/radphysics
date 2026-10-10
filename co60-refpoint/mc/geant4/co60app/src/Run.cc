#include "Run.hh"

Run::Run() : tallies(Config::Get()) {}

void Run::Merge(const G4Run* other)
{
  tallies.Merge(static_cast<const Run*>(other)->tallies);
  G4Run::Merge(other);
}
