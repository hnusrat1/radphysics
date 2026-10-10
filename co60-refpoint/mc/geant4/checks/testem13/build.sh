#!/usr/bin/env bash
# Build TestEm13 (transmission measurement of photon cross sections) with an added
# "emstandard_opt4" choice, so the tracking cross sections of the production physics
# list can be measured directly. The example is copied, patched, and built out of tree.
set -euo pipefail
G4=/opt/geant4/11.4.3
SRC=$G4/share/Geant4/examples/extended/electromagnetic/TestEm13
WORK=/opt/build/examples/TestEm13-opt4
rm -rf "$WORK" && mkdir -p "$WORK" && cp -r "$SRC" "$WORK/src"
PL="$WORK/src/src/PhysicsList.cc"
sed -i 's|#include "PhysListEmStandard.hh"|#include "PhysListEmStandard.hh"\n#include "G4EmStandardPhysics_option4.hh"|' "$PL"
sed -i 's|  else if (name == "livermore") {|  else if (name == "emstandard_opt4") {\n    fEmName = name;\n    delete fEmPhysicsList;\n    fEmPhysicsList = new G4EmStandardPhysics_option4();\n  }\n  else if (name == "livermore") {|' "$PL"
grep -q 'G4EmStandardPhysics_option4()' "$PL"
cmake -S "$WORK/src" -B "$WORK/build" -DGeant4_DIR=$G4/lib/cmake/Geant4 -DCMAKE_BUILD_TYPE=Release > "$WORK/cmake.log"
cmake --build "$WORK/build" -j 2 > "$WORK/make.log"
echo "built $WORK/build/TestEm13"
