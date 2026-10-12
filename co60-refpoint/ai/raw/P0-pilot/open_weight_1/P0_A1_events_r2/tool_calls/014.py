import math
m_e = 0.51099895
r_e_sq = 7.9408e-26
n_e = 3.3428e23

def mu_Compton(E):
    alpha = E / m_e
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - math.log(1+2*alpha)/alpha)
    t2 = math.log(1+2*alpha) / (2*alpha)
    t3 = (1+3*alpha) / (1+2*alpha)**2
    sigma = 2*math.pi * r_e_sq * (t1 + t2 - t3)
    return n_e * sigma

def mu_Rayleigh(E):
    return 0.0008 * (1.0 / max(E, 0.01))**2

def mu_Photoelectric(E):
    return 1.9e-5 * (1.0 / max(E, 0.01))**3.5

E = 0.2266
print(f"At E={E:.4f} MeV:")
print(f"  mu_C = {mu_Compton(E):.6f}")
print(f"  mu_R = {mu_Rayleigh(E):.6f}")
print(f"  mu_P = {mu_Photoelectric(E):.6f}")
print(f"  Total = {mu_Compton(E)+mu_Rayleigh(E)+mu_Photoelectric(E):.6f}")
print(f"  Rayleigh fraction = {mu_Rayleigh(E)/(mu_Compton(E)+mu_Rayleigh(E)+mu_Photoelectric(E)):.4f}")
print(f"  Photo fraction = {mu_Photoelectric(E)/(mu_Compton(E)+mu_Rayleigh(E)+mu_Photoelectric(E)):.4f}")