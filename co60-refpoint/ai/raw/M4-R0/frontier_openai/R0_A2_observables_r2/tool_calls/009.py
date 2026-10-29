e=np.loadtxt('/tmp/data/estar_water.txt',comments='#');
for E in [.1,.2,.5,1,1.2,1.4]:
 i=np.argmin(abs(e[:,0]-E));print(e[i,0],e[i,1:6])