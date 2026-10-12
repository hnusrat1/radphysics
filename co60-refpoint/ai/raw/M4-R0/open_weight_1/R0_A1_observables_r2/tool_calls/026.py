import sys, os
sys.path.insert(0, os.getcwd())
src = open('mc_photon.py').read()

# add depth_tr init
src = src.replace("    E_tr_scat=0.0\n    el_ini_all=", "    E_tr_scat=0.0\n    depth_tr = np.zeros(150)\n    el_ini_all=")

# add depth recording for compton
src = src.replace("                E_tr_scat += Tsub[~pc].sum()\n        if coh_a.any():", "                E_tr_scat += Tsub[~pc].sum()\n                # depth\n                zb = np.clip((p1[ic][:,2]/0.2).astype(int),0,149)\n                np.add.at(depth_tr, zb, Tsub)\n        if coh_a.any():")
# photo depth
src = src.replace("                E_tr_scat+=Tsub[ic][~pc].sum()\n            alive[act_idx[pho_a]]=False", "                E_tr_scat+=Tsub[ic][~pc].sum()\n                zb = np.clip((p1[ic][:,2]/0.2).astype(int),0,149)\n                np.add.at(depth_tr, zb, Tsub[ic])\n            alive[act_idx[pho_a]]=False")
# pair depth
src = src.replace("                E_tr_scat+=tr[~pc].sum()\n            alive[act_idx[pri_a]]=False", "                E_tr_scat+=tr[~pc].sum()\n                zb = np.clip((p1[ic][:,2]/0.2).astype(int),0,149)\n                np.add.at(depth_tr, zb, tr)\n            alive[act_idx[pri_a]]=False")
# return depth_tr
src = src.replace("                n_el_pri=n_el_pri, N_el_e=N_el_e, N_el_pos=N_el_pos,\n                N=N, t=time.time()-t0)", "                n_el_pri=n_el_pri, N_el_e=N_el_e, N_el_pos=N_el_pos,\n                depth_tr=depth_tr,\n                N=N, t=time.time()-t0)")
open('mc_photon.py','w').write(src)
import importlib, mc_photon
importlib.reload(mc_photon)
print("patched ok")