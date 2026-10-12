import numpy as np

# ============================================================
# Cross sections and interaction sampling
# ============================================================

m_e_c2 = 0.510998950
NA = 6.02214076e23
e_per_cm3 = 10 * NA / 18.01528

def get_mu_components(E):
    """Return [mu_C, mu_P, mu_PP, mu_R] in cm^-1"""
    comp_E = np.array([0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0, 3.0, 5.0])
    # Compton (incoherent) mass attenuation - NIST-based for water
    comp_murho = np.array([0.151, 0.140, 0.130, 0.119, 0.100, 0.084, 0.077, 0.069, 0.069, 0.071, 0.073, 0.073, 0.072, 0.071, 0.0585, 0.0575, 0.0550, 0.0522, 0.0412, 0.0350, 0.0290])
    # Actually let me be more careful. The Compton coefficient decreases with energy
    # Let me use the KN formula instead - it's more accurate
    
    # Let me use the Klein-Nishina directly
    alpha = E / m_e_c2
    t1 = (1+alpha)/alpha**2 * (2*(1+alpha)/(1+2*alpha) - np.log(1+2*alpha)/alpha)
    t2 = np.log(1+2*alpha)/(2*alpha)
    t3 = (1+3*alpha)/(1+2*alpha)**2
    sigma_KN = 2*np.pi*(2.8179403262e-13)**2*(t1 + t2 - t3)
    mu_c = sigma_KN * e_per_cm3
    
    # Photoelectric - NIST data for water
    photo_E = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1732, 1.25, 1.3325, 1.5, 2.0])
    photo_murho = np.array([4.91, 1.67, 0.755, 0.216, 0.0881, 0.0442, 0.0255, 0.0113, 0.00642, 0.00254, 0.00222, 0.00322, 0.00406, 0.00555, 0.0045, 0.00377, 0.0030, 0.00258, 0.0023, 0.00179, 0.0010])
    # Hmm wait, photoelectric decreases with energy but I see it increasing from 0.3 to 0.5... that's wrong
    # Let me fix the data
    pass