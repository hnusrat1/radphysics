#ifndef CO60_MESSENGER_HH
#define CO60_MESSENGER_HH

#include "G4UImessenger.hh"

#include <map>

class G4UIcommand;
class G4UIdirectory;

// UI commands under /co60/. None is broadcast to worker threads (see Config.hh).
class Messenger : public G4UImessenger
{
 public:
  Messenger();
  ~Messenger() override;
  void SetNewValue(G4UIcommand*, G4String) override;

 private:
  G4UIcommand* Make(const G4String& path, const G4String& guidance, const G4String& params);
  std::map<G4String, G4UIcommand*> fCommands;
  G4UIdirectory* fDirs[7];
};

#endif
