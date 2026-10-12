
import subprocess,time,pickle
t=time.time()
r=subprocess.run(['python','mc.py','2','2000000'],capture_output=True,text=True)
print(r.stdout[-300:],r.stderr[-1000:],time.time()-t)
o=pickle.load(open('out_2.pkl','rb'))
NORM=o['NORM']
print("primary fluence/phi0", NORM*o['Lsum']['P'], "scattered", NORM*o['Lsum']['S'])
print("analytic primary", np.exp(-0.04942*10))
