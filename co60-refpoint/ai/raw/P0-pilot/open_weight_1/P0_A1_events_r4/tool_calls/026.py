def simulate_one_fast(rng):
    events = []
    u = rng.uniform(-FIELD_HALF, FIELD_HALF)
    v = rng.uniform(-FIELD_HALF, FIELD_HALF)
    inv_norm = 1.0 / sqrt(u*u + v*v + 10000.0)
    dx = u * inv_norm
    dy = v * inv_norm
    dz = 100.0 * inv_norm
    px = u
    py = v
    pz = Z_SURFACE
    energy = 2.0
    for _ in range(200):
        mu_t, mu_c, mu_pe, mu_pp, mu_r = mu_total_fast(energy)
        if mu_t <= 0:
            break
        mfp = 1.0 / mu_t
        s = -mfp * log(1.0 - rng.random())
        new_px = px + s * dx
        new_py = py + s * dy
        new_pz = pz + s * dz
        if new_pz >= Z_EXIT or new_pz < Z_SURFACE:
            break
        in_scoring = (new_pz >= SV_Z_MIN and new_pz <= SV_Z_MAX and new_px*new_px + new_py*new_py <= SV_RADIUS_SQ)
        xi = rng.random()
        p_c = mu_c / mu_t
        p_pe = mu_pe / mu_t
        p_pp = mu_pp / mu_t
        e_in = energy
        e_out = None
        cos_scatter = None
        t_e = 0.0
        cos_electron = None
        t_pos = None
        absorbed = False
        if xi < p_c:
            process = 'compton'
            e_out, cos_scatter, t_e, cos_electron = sample_compton_fast(energy, rng)
            azimuth = rng.random() * 2*pi
            dx, dy, dz = rotate_dir_fast(dx, dy, dz, cos_scatter, azimuth)
            energy = e_out
        elif xi < p_c + p_pe:
            process = 'photoelectric'
            t_e = energy
            cos_electron = rng.uniform(-1.0, 1.0)
            energy = 0.0
            absorbed = True
        elif xi < p_c + p_pe + p_pp:
            process = 'pair'
            T_total = energy - 2*m_e
            if T_total > 0:
                t_e = rng.uniform(0.0, T_total)
                t_pos = T_total - t_e
            else:
                t_e = 0.0
                t_pos = 0.0
            theta_avg = m_e / energy if energy > 0 else 0
            if theta_avg > 0:
                theta = theta_avg * sqrt(-2.0 * log(1.0 - rng.random()))
                cos_electron = cos(theta)
            else:
                cos_electron = 1.0
            energy = 0.0
            absorbed = True
        else:
            process = 'rayleigh'
            Z_eff = 10.0
            theta0 = 0.02 / (max(energy, 0.01) * Z_eff**(1/3))
            theta = theta0 * sqrt(-2.0 * log(1.0 - rng.random()))
            cos_scatter = cos(theta)
            e_out = energy
            azimuth = rng.random() * 2*pi
            dx, dy, dz = rotate_dir_fast(dx, dy, dz, cos_scatter, azimuth)
            cos_electron = None
            t_e = 0.0
        if in_scoring:
            events.append({'process': process, 'e_in_MeV': float(e_in), 'e_out_MeV': float(e_out) if e_out is not None else None, 'cos_scatter': float(cos_scatter) if cos_scatter is not None else None, 't_e_MeV': float(t_e), 'cos_electron': float(cos_electron) if cos_electron is not None else None, 't_positron_MeV': float(t_pos) if t_pos is not None else None})
        px, py, pz = new_px, new_py, new_pz
        if absorbed:
            break
    return events