for E in [0.2, 0.3, 0.5]:
    mt, mc, mr, mp, mpr = partial_mu(E)
    print(f"E={E}: tot={mt:.4f}, comp={mc:.4f}, rayl={mr:.4f}, pe={mp:.4f}, pair={mpr:.4f}")
    print(f"  fractions: comp={mc/mt*100:.1f}%, rayl={mr/mt*100:.1f}%, pe={mp/mt*100:.1f}%")