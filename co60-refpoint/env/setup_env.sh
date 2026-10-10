# Source this file to set up Geant4, EGSnrc, and the Python environment.
# Override the locations with G4_PREFIX, EGS_ROOT, VENV before sourcing.
G4_PREFIX=${G4_PREFIX:-/opt/geant4/11.4.3}
EGS_ROOT=${EGS_ROOT:-/opt/EGSnrc}
VENV=${VENV:-/opt/venv}

# geant4.sh locates itself from BASH_SOURCE; run it from its own directory.
pushd "$G4_PREFIX/bin" >/dev/null && source ./geant4.sh && popd >/dev/null
[ -f "$EGS_ROOT/egsnrc_env.sh" ] && source "$EGS_ROOT/egsnrc_env.sh"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
