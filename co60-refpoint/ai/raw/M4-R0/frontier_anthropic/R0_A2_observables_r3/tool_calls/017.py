
s = open('/tmp/work/ana.py').read()
s = s.replace("Phi_tot = solve_phi(Q)\n", "Phi_tot = solve_phi(Q)\nprint('DIAG births delta per Phi0 (per vol):', np.sum(Kmat@Phi_tot*dT), ' sum Qd dT', np.sum(Qd_new*dT) if False else np.sum((Kmat@Phi_tot)*dT), ' L4 solver', np.sum(Phi_tot*dT), 'Phi_d sum', np.sum((Phi_tot-solve_phi(Q0))*dT))\n")
s = s.replace("Yn = interpE(Te,Yrad,allT) if False else np.interp(np.log(allT), np.log(Te), Yrad)","Yn = np.interp(np.log(np.maximum(allT,0.01)), np.log(Te), Yrad)")
s = s.replace("Kcol_s = np.sum(sb_w*sb_T*(1-np.interp(np.log(sb_T), np.log(Te), Yrad)))/V*CONV","Kcol_s = np.sum(sb_w*sb_T*(1-np.interp(np.log(np.maximum(sb_T,0.01)), np.log(Te), Yrad)))/V*CONV")
s = s.replace("Kcol_p = np.sum(pb_w*pb_T*(1-np.interp(np.log(pb_T), np.log(Te), Yrad)))/V*CONV","Kcol_p = np.sum(pb_w*pb_T*(1-np.interp(np.log(np.maximum(pb_T,0.01)), np.log(Te), Yrad)))/V*CONV")
open('/tmp/work/ana.py','w').write(s)
import subprocess
r=subprocess.run("cd /tmp/work && python ana.py | head -5",shell=True,capture_output=True,text=True)
print(r.stdout, r.stderr[-1500:])
