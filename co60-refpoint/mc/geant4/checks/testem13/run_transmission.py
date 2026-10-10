"""Measure the tracking attenuation coefficient of G4_WATER at the Co-60 lines (TestEm13, option4).

The fraction of photons that cross a water slab without any interaction is
exp(-mu x) with the cross sections Geant4 actually uses in tracking. This is
compared with G4EmCalculator::ComputeCrossSectionPerVolume, the source of the
analytic primary fluence in analysis/primary.py.

Usage: python mc/geant4/checks/testem13/run_transmission.py <run_id> <n_per_line>
"""

import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXE = "/opt/build/examples/TestEm13-opt4/build/TestEm13"
LINES = {"line1": 1.1732, "line2": 1.3325}
THICK_CM = 5.0


def main(run_id, n):
    out = ROOT / "mc" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    tmpl = (HERE / "transmission.mac.in").read_text()
    res = {"run_id": run_id, "executable": EXE, "thickness_cm": THICK_CM, "material": "G4_WATER",
           "physics": "G4EmStandardPhysics_option4, UseGeneralProcess false", "lines": {}}
    for i, (name, e) in enumerate(LINES.items()):
        mac = tmpl.replace("@THICK@", str(THICK_CM)).replace("@E@", str(e)).replace("@N@", str(n))
        mac = mac.replace("@SEED1@", str(7001 + 2 * i)).replace("@SEED2@", str(7002 + 2 * i))
        (out / f"{name}.mac").write_text(mac)
        log = out / f"{name}.log"
        with open(log, "w") as fh:
            subprocess.run(["nice", "-n", "10", EXE, str(out / f"{name}.mac"), "1"], stdout=fh,
                           stderr=subprocess.STDOUT, check=True, cwd=out)
        txt = log.read_text()
        m = re.search(r"unaltered after .*? : (\d+) over (\d+) incident", txt)
        survive, total = int(m.group(1)), int(m.group(2))
        p = survive / total
        mu = -math.log(p) / THICK_CM
        mu_sem = math.sqrt((1 - p) / (p * total)) / THICK_CM
        calc = re.search(r"Verification from G4EmCalculator: \n(.*?)\n", txt).group(1)
        calc_terms = {k: v for k, v in re.findall(r"(\w+)= ([\d.e+-]+ \S+)", calc)}
        res["lines"][name] = {"energy_MeV": e, "transmitted": survive, "incident": total,
                              "mu_measured_cm-1": mu, "mu_measured_sem_cm-1": mu_sem,
                              "calculator_terms_as_printed": calc_terms}
    yaml.safe_dump(res, open(out / "results.yaml", "w"), sort_keys=False)
    print(yaml.safe_dump(res, sort_keys=False))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
