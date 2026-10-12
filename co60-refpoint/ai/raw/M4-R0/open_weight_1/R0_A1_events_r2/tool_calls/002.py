import numpy as np

r_e = 2.8179403262e-13  # cm
mec2 = 0.5109989461  # MeV

def klein_nishina_total(energy_MeV):
    alpha = energy_MeV / mec2
    term = ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
            + np.log(1+2*alpha)/(2*alpha) 
            - (1+3*alpha)/(1+2*alpha)**2)
    sigma = 2 * np.pi * r_e**2 * term
    return sigma

# test at alpha=0.001 (very low energy)
alpha = 0.001
term = ((1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha) 
        + np.log(1+2*alpha)/(2*alpha) 
        - (1+3*alpha)/(1+2*alpha)**2)
print("term:", term)
print("expected term for Thomson: 4/3 =", 4/3)
sigma = 2 * np.pi * r_e**2 * term
print("sigma:", sigma)
# Thomson cross section
sigma_thomson = 8/3 * np.pi * r_e**2
print("Thomson sigma:", sigma_thomson)
print("ratio:", sigma/sigma_thomson)