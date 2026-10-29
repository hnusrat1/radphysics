code = open('/tmp/photonmc.py').read()
# add pd_depths parameter and tallies
old = "def photon_mc(N, seed=42, region=4.5, region_zmax=13.0, save_src=True, verbose=True, debug=False):"
new = "def photon_mc(N, seed=42, region=4.5, region_zmax=13.0, save_src=True, verbose=True, debug=False, pd_depths=None, pd_half=0.1, pd_r=1.0):"
assert old in code; code=code.replace(old,new)

old = "    n_cyl_seg_prim=0; n_cyl_seg_scat=0"
new = "    n_cyl_seg_prim=0; n_cyl_seg_scat=0\n    if pd_depths is not None:\n        pd_depths=np.asarray(pd_depths,float)\n        Kdep=np.zeros(len(pd_depths)); Ndep=np.zeros(len(pd_depths))\n    else:\n        pd_depths=None; Kdep=None; Ndep=None"
assert old in code; code=code.replace(old,new)

# insert depth tally inside `if in_sc.any():` block? Actually tally Etr for all interacting photons in each depth cylinder.
# We'll add after the Layer3/etr computation, before `if in_sc.any(): ... int_all` block.
old = """            if in_sc.any():
                psub = proc[in_sc]; prsub = prim_part[in_sc]
                np.add.at(int_all, psub, 1)
                np.add.at(int_prim, psub[prsub], 1)
                np.add.at(int_scat, psub[~prsub], 1)"""
new = """            if in_sc.any():
                psub = proc[in_sc]; prsub = prim_part[in_sc]
                np.add.at(int_all, psub, 1)
                np.add.at(int_prim, psub[prsub], 1)
                np.add.at(int_scat, psub[~prsub], 1)
            # PDD depth kerma tally: Etr per depth
            if pd_depths is not None:
                Etr = np.zeros(len(ii))
                mm = m_inc
                if mm.any():
                    # recompute Te for all compton (we have them only for m_inc subset); we stored Te wrt subset
                    pass
                # we recompute Compton Te for all interactions cheaply here
                # (already have newE for compton rows; Etr = E_int - newE for compton)
                Etr[m_inc] = E_int[m_inc]-newE[m_inc]
                Etr[m_pe] = E_int[m_pe]
                Etr[m_pair] = np.maximum(E_int[m_pair]-1.022,0)
                for kd,zd in enumerate(pd_depths):
                    in_d = (np.abs(p_int[:,2]-zd)<=pd_half)&(p_int[:,0]**2+p_int[:,1]**2<=pd_r**2)
                    if in_d.any():
                        Kdep[kd]+= Etr[in_d].sum()
                        Ndep[kd]+= in_d.sum()"""
assert old in code; code=code.replace(old,new)

# add Kdep to result
old = "        n_cyl_seg_prim=n_cyl_seg_prim, n_cyl_seg_scat=n_cyl_seg_scat"
new = "        n_cyl_seg_prim=n_cyl_seg_prim, n_cyl_seg_scat=n_cyl_seg_scat, Kdep=Kdep, Ndep=Ndep, pd_depths=pd_depths"
assert old in code; code=code.replace(old,new)
open('/tmp/photonmc.py','w').write(code)
print('patched')