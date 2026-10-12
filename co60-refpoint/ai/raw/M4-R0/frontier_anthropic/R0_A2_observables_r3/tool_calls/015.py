
src = open('/tmp/work/ana.py').read()
src = src.replace("R = pickle.load(open('/tmp/work/res_t1.pkl','rb'))",
"R = pickle.load(open('/tmp/work/res_t1.pkl','rb'))\nFB = 1.25e6/R['N']\nfor _k in ['inc','pe','pair','coh']:\n    R['cnt'][_k]*=FB; R['etr'][_k]*=FB\nR['Esum_w']*=FB")
src = src.replace("sb_T=R['birth_T']; sb_w=R['birth_w']","sb_T=R['birth_T']; sb_w=R['birth_w']*FB")
src = src.replace("prod = Phi_tot*dT*np.array([sigma_delta_above(E) for E in Tnode])*NEL/NEL",
"prod = Phi_tot*dT*np.array([sigma_delta_above(E) if E>0.02 else 0.0 for E in Tnode])")
src = src.replace("def sigma_delta_above(Ep, D=0.01):","def sigma_delta_above(Ep, D=0.01):\n    if Ep<=2*D: return 0.0")
src = src.replace("Esum_s = R['Esum_w']/V","Esum_s = R['Esum_w']/V")
open('/tmp/work/ana.py','w').write(src)
import subprocess
r=subprocess.run("cd /tmp/work && python ana.py",shell=True,capture_output=True,text=True)
print(r.stdout[-9000:], r.stderr[-3000:])
