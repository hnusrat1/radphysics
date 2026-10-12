
for key in ['L1_scattered_fraction_fluence','L1_scattered_fraction_energy_fluence','L5_dose_fraction_parent_scattered','L5_K_over_phi0_Gy_cm2','L5_s_w_air','L4_delta_fraction','L2_scattered_share_of_interactions','L4_electron_fluence_per_phi0']:
    v=np.array([o[key] for o in outs]); print(key, np.round(v,5), 'mean',v.mean().round(5),'sem',(v.std(ddof=1)/np.sqrt(len(v))).round(5))
