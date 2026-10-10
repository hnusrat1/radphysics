#!/usr/bin/env bash
# Build the pinned Geant4 release used by this study.
#
# Usage:
#   PREFIX=/opt/geant4 ./install_geant4.sh            # data from CERN (CMake verifies MD5)
#   PREFIX=/opt/geant4 DATA_SOURCE=conda ./install_geant4.sh
#
# DATA_SOURCE=conda installs the data libraries from conda-forge packages that
# repackage the same CERN tarballs. Use it only where geant4-data.web.cern.ch is
# unreachable. Package hashes are pinned in versions.lock.yaml.
set -euo pipefail

G4_TAG=v11.4.3
G4_COMMIT=2ee379e6843e55b9c6bbb835d2a2b03cc998fef0
PREFIX=${PREFIX:-/opt/geant4}
SRC=${SRC:-$PREFIX/src/geant4-$G4_TAG}
BUILD=${BUILD:-$PREFIX/build/geant4-$G4_TAG}
INSTALL=$PREFIX/${G4_TAG#v}
DATADIR=$PREFIX/data
DATA_SOURCE=${DATA_SOURCE:-cern}
NCPU=${NCPU:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$G4_TAG" https://github.com/Geant4/geant4.git "$SRC"
fi
actual=$(git -C "$SRC" rev-parse HEAD)
if [ "$actual" != "$G4_COMMIT" ]; then
  echo "Geant4 source commit $actual does not match pinned $G4_COMMIT" >&2
  exit 1
fi

install_data=ON
[ "$DATA_SOURCE" = conda ] && install_data=OFF

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEANT4_BUILD_MULTITHREADED=ON \
  -DGEANT4_INSTALL_DATA="$install_data" \
  -DGEANT4_INSTALL_DATADIR="$DATADIR" \
  -DGEANT4_USE_SYSTEM_EXPAT=ON \
  -DGEANT4_USE_QT=OFF -DGEANT4_USE_OPENGL_X11=OFF -DGEANT4_USE_GDML=OFF \
  -DGEANT4_INSTALL_EXAMPLES=ON
cmake --build "$BUILD" -j "$NCPU"
cmake --install "$BUILD"

if [ "$DATA_SOURCE" = conda ]; then
  "$(dirname "$0")/install_geant4_data_conda.sh" "$DATADIR"
fi

echo "Geant4 installed in $INSTALL; source $INSTALL/bin/geant4.sh"
