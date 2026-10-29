import numpy as np

r_e = 2.8179403205e-13
mec2 = 0.5109989461

def klein_nishina_spectrum(E_in, n=100000):
    """Sample n Compton scattering events and compute average outgoing energy"""
    alpha = E_in / mec2
    # Sample cos(theta) using Kahn's method or direct inversion
    # We'll use the rejection method
    cos = []
    for _ in range(n):
        while True:
            # Sample mu uniformly in [-1,1]
            mu = np.random.uniform(-1, 1)
            # Compute Klein-Nishina DCS (proportional)
            k = alpha
            f = 1 / (1 + k*(1-mu))
            dcs = f**2 * (f + 1/f - (1-mu**2)) / 2  # normalized to Thomson
            # Rejection: compare to maximum value
            # Max DCS is at mu=1? Actually at forward scattering,
            # f=1, dcs = 1 + 1 - 0 = 2. So max = 2.
            if np.random.ranom() < dcs / 2:
                cos. append(mu)
                break
    # Compute outgoing energy
    E_out = E_in / (1 + alpha*(1-np.array(cos)))
    return np.mean(E_out)

avg_Eout = klein_nishina_spectrum(1.1732, 50000)
print("Average outgoing energy:", avg_Eout)
print("Average energy transfer:", 1.1732 - avg_Eout)