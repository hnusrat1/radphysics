#!/usr/bin/env bash
# Install the pinned EGSnrc release and compile the RZ user codes used for the
# B0 benchmark (DOSRZnrc, FLURZnrc, SPRRZnrc) plus tutor1 as a smoke test.
#
# Usage: PREFIX=/opt ./install_egsnrc.sh
# Needs: git, gfortran, gcc, g++, make, expect.
set -euo pipefail

EGS_TAG=v2026
EGS_COMMIT=bd49a47aae3938f49f4100b8e018091861408b19
PREFIX=${PREFIX:-/opt}
ROOT=$PREFIX/EGSnrc
CONF=${CONF:-linux}

if [ ! -d "$ROOT" ]; then
  git clone https://github.com/nrc-cnrc/EGSnrc.git "$ROOT"
fi
git -C "$ROOT" checkout -q "$EGS_TAG"
actual=$(git -C "$ROOT" rev-parse HEAD)
if [ "$actual" != "$EGS_COMMIT" ]; then
  echo "EGSnrc commit $actual does not match pinned $EGS_COMMIT" >&2
  exit 1
fi

cd "$ROOT"
# Argument 3: compile no user codes here; the ones needed are compiled below.
HEN_HOUSE/scripts/configure.expect "$CONF" 3

cat > "$ROOT/egsnrc_env.sh" <<EOF
export HEN_HOUSE=$ROOT/HEN_HOUSE/
export EGS_HOME=$ROOT/egs_home/
export EGS_CONFIG=$ROOT/HEN_HOUSE/specs/$CONF.conf
source $ROOT/HEN_HOUSE/scripts/egsnrc_bashrc_additions
EOF

# shellcheck disable=SC1091
source "$ROOT/egsnrc_env.sh"
for code in tutor1 dosrznrc flurznrc sprrznrc; do
  (cd "$EGS_HOME/$code" && make)
done
echo "EGSnrc installed; source $ROOT/egsnrc_env.sh"
