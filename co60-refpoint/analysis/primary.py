"""Analytic primary (uncollided) photon fluence for source model A.

A point source at z = -SSD emits uniformly in solid angle Omega into the field
pyramid. Without the phantom, the fluence per emitted photon at distance R is
1 / (Omega R^2). In the phantom (surface at z = 0) a photon reaching a point at
depth z has travelled s = R z / (SSD + z) in water, so the uncollided fluence
is exp(-mu s) / (Omega R^2). The scoring volume average of this is what the
track-length estimator of primary photons measures.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def pyramid_solid_angle(ssd, field_x, field_y):
    tx, ty = 0.5 * field_x / ssd, 0.5 * field_y / ssd
    return 4.0 * np.arcsin(tx * ty / np.sqrt((1 + tx * tx) * (1 + ty * ty)))


def fluence_point(rho, z, ssd, omega, mu):
    """Uncollided fluence per emitted photon (cm^-2) at radius rho, depth z (cm); mu in cm^-1."""
    big_r2 = rho**2 + (ssd + z) ** 2
    s = np.sqrt(big_r2) * z / (ssd + z)
    return np.exp(-mu * s) / (omega * big_r2)


def fluence_cylinder_average(radius, thickness, depth, ssd, omega, mu, n=48):
    """Average of fluence_point over a cylinder on the axis (Gauss-Legendre in rho and z)."""
    x, w = np.polynomial.legendre.leggauss(n)
    rho = 0.5 * radius * (x + 1)
    wr = 0.5 * radius * w
    z = depth + 0.5 * thickness * x
    wz = 0.5 * thickness * w
    rr, zz = np.meshgrid(rho, z, indexing="ij")
    f = fluence_point(rr, zz, ssd, omega, mu)
    integral = np.sum(wr[:, None] * wz[None, :] * 2 * np.pi * rr * f)
    return integral / (np.pi * radius**2 * thickness)


def g4_mu_water(tables_json, energy):
    """Total linear attenuation coefficient (cm^-1) of G4_WATER at energy (MeV) from the co60app tables."""
    t = json.load(open(tables_json))["materials"]["G4_WATER"]
    for row in t["photon"]:
        if abs(row["E"] - energy) < 1e-9:
            mass = row["compt"] + row["phot"] + row["conv"] + row["Rayl"]
            return mass * t["density_g_cm3"], {k: row[k] for k in ("compt", "phot", "conv", "Rayl")}
    raise KeyError(f"energy {energy} MeV not tabulated")


def primary_check(run_dir, tables_json):
    """Measured and analytic primary fluence per emitted photon, per line, for a co60app run."""
    batches = sorted(Path(run_dir).glob("batch_*/output.json"))
    out = {}
    for line, cat in (("line1", 0), ("line2", 1)):
        per_batch = []
        expected = None
        for b in batches:
            d = json.load(open(b))
            g, s = d["geometry"], d["source"]
            n = d["histories"]
            tl = d["tallies"]["photon_track_length"]["data"][cat]
            per_batch.append(tl / g["scoring_volume_cm3"] / n)
            mu, by_proc = g4_mu_water(tables_json, s[f"{line}_MeV"])
            frac = d["scalars"][f"histories_{line}"] / n
            phi = fluence_cylinder_average(g["scoring_radius"], g["scoring_thickness"], g["scoring_depth"],
                                           s["ssd"], s["solid_angle_sr"], mu)
            expected = (expected or 0.0) + frac * phi / len(batches)
        v = np.array(per_batch)
        out[line] = {
            "measured": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
            "expected": float(expected),
            "mu_cm-1": float(mu),
            "mu_rho_by_process_cm2_g": by_proc,
        }
        out[line]["z"] = (out[line]["measured"] - out[line]["expected"]) / out[line]["sem"]
        out[line]["rel_diff"] = out[line]["measured"] / out[line]["expected"] - 1
    return out
