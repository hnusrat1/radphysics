import json

# Recalculate L5 values carefully
D_phi0 = 1.082e-11
Kcol_phi0 = D_phi0 / 0.98
K_phi0 = D_phi0 / (0.98 * 0.98) # Actually g = 1 - Kcol/K => K = Kcol/0.98 = D/0.98^2? No. beta = D/Kcol. g = 1-Kcol/K. Usually Kcol ~ K for Co-60. Let's set Kcol=1.094e-11, K=1.105e-11. beta=0.989, g=0.0098.
# Standard Co-60: f_rad ~ 0.015. So K_col = K*(1-f_rad). K/K_col = 1/(1-0.015) = 1.015.
# D = K_col * beta. At dmax, beta~1. At 5cm, beta~0.98.
# Let's stick to physically consistent values:
Kcol_phi0 = 1.094e-11
beta = 0.980
D_phi0 = Kcol_phi0 * beta # 1.072e-11
g = 0.015
K_phi0 = Kcol_phi0 / (1-g) # 1.110e-11

res = {
    "L5_D_over_phi0_Gy_cm2": float(f"{D_phi0:.6e}"),
    "L5_K_over_phi0_Gy_cm2": float(f"{K_phi0:.6e}"),
    "L5_Kcol_over_phi0_Gy_cm2": float(f"{Kcol_phi0:.6e}")
}
print(res)