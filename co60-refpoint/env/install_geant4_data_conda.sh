#!/usr/bin/env bash
# Install the Geant4 11.4.3 data libraries from conda-forge noarch packages.
# Each package repackages the official CERN tarball; the tarball sha256 is in
# the package recipe and copied into versions.lock.yaml. The package sha256
# values below are checked before extraction.
#
# Usage: ./install_geant4_data_conda.sh <datadir>
# Needs: curl, python3, zstd, tar, sha256sum (or shasum on macOS).
set -euo pipefail

DATADIR=${1:?usage: $0 <datadir>}
CHANNEL=https://conda.anaconda.org/conda-forge/noarch
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# package file, package sha256, directory name inside the package, Geant4 directory name
PKGS="
geant4-data-emlow-8.8.0-hd8ed1ab_0.conda cd5cc7031be23c3a296fc824616091696fac82e7022df8d3c118c190e794b8e0 EMLOW8.8 G4EMLOW8.8
geant4-data-ensdfstate-3.0-hd8ed1ab_0.conda 93858ea8dcce9dd0d25d59fd311052bf29763498cdba1c687b1934b082917f45 ENSDFSTATE3.0 G4ENSDFSTATE3.0
geant4-data-particlexs-4.2.0-hd8ed1ab_0.conda 858b333fb96b3575bf20df7c2e1fe48a70968dc35696eb1686c8c69614760c45 PARTICLEXS4.2 G4PARTICLEXS4.2
geant4-data-photonevaporation-6.1.2-hd8ed1ab_0.conda 2e5f496a370fe9ec27e874bba8be9c61cfa44a289178fbf02bdb0af55f476403 PhotonEvaporation6.1.2 PhotonEvaporation6.1.2
geant4-data-radioactivedecay-6.1.2-hd8ed1ab_0.conda 0928ad4205286a1b7548c4fecb0301fd94d0bc3fa77f742e7fb0bef3d0809799 RadioactiveDecay6.1.2 RadioactiveDecay6.1.2
geant4-data-abla-3.3-hd8ed1ab_1.conda 68bfb8773efa34e94ac0b8b7d19379a5de5fe46684b61d4b1ee883d2063df5c8 ABLA3.3 G4ABLA3.3
geant4-data-incl-1.3.0-hd8ed1ab_1.conda 60daadcd66f20a5fbce97b7d76b9448fb53dfb7f8114e007f9cd5390b74ef35d INCL1.3 G4INCL1.3
geant4-data-pii-1.3-hd8ed1ab_2.conda 00a097687383079ccc4f6c4b25dc9888cd80492f32ef55d4e01f8c54efcc68fe PII1.3 G4PII1.3
geant4-data-saiddata-2.0-hd8ed1ab_1.conda d7bee123a2223284882708e8796d840fb975c81532631983561eea543ebbaa20 SAIDDATA2.0 G4SAIDDATA2.0
"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

mkdir -p "$DATADIR"
echo "$PKGS" | while read -r file hash inner target; do
  [ -z "$file" ] && continue
  if [ -d "$DATADIR/$target" ]; then echo "exists: $target"; continue; fi
  curl -sS --retry 3 -o "$WORK/$file" "$CHANNEL/$file"
  got=$(sha256 "$WORK/$file")
  if [ "$got" != "$hash" ]; then echo "sha256 mismatch for $file: $got" >&2; exit 1; fi
  mkdir -p "$WORK/x"
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$WORK/$file" "$WORK/x"
  for t in "$WORK"/x/pkg-*.tar.zst; do zstd -dc "$t" | tar -x -C "$WORK/x"; done
  mv "$WORK/x/share/Geant4/data/$inner" "$DATADIR/$target"
  rm -rf "$WORK/x" "$WORK/$file"
  echo "installed: $target"
done
