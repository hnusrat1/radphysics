import numpy as np
Eprim=.366*1.1732+.369*1.3325
w=np.array([.001,.002,.003,.004,.006,.009,.012,.016,.02,.025,.03,.035,.04,.043,.045,.045,.044,.042,.04,.038,.035,.03,.024,.017,.009,.003,.0005,.0001]); cen=np.array([.025+i*.05 for i in range(28)]); mean=w@cen/w.sum(); esc=.3*mean
print(Eprim,mean,esc,(Eprim+esc)/1.035,esc/(Eprim+esc))