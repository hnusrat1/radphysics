import numpy as np

r_e = 2.8179403205e-13
mec2 = 0.5109989461
np.random.seed(12345)

def avg_energy_transfer(E_in, n=100000):
    alpha = E_in / mec2
    cos_list = []
    for _ in range(n):
        while True:
            mu = np.random.uniform(-1, 1)
            k = alpha
            f = 1 / (1 + k*(1-mu))
            dcs = f**2 * (f + 1/f - (1-mu**2)) / 2  # normalized such that max=2 at mu=1
            if np.random.random() < dcs / 2:
                cos_list.append(mu)
                break
    cos = np.array(cos_list)
    E_out = E_in / (1 + alpha*(1-cos))
    avg_E_out = np.mean(E_out)
    avg_E_transfer = E_in - avg_E_out
    return avg_E_transfer, avg_E_out

trans, out = avg_energy_transfer(1.1732, 200000)
print("Avg energy transfer:", trans, "MeV")
print("Avg outgoing energy:", out, "MeV")