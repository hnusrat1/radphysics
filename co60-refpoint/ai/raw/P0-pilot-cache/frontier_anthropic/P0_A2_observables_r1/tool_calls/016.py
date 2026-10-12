
import numpy as np, sys; sys.path.insert(0,'/tmp/work'); import xs
RE=2.8179403e-13; ME=0.51099895; NA=6.02214076e23
NE_w=NA*(1.0*(2*1+8)/18.0)*1.0  # electrons per gram water: NA*Z/A, Z/A=10/18
NE_w=NA*0.55509
def Fx(x,tau):
    return 1/x**2+1/(1-x)**2+(tau/(tau+1))**2-(2*tau+1)/((tau+1)**2*x*(1-x))
def beta2(T):
    g=1+T/ME; return 1-1/g**2
def Scol_moller(T,I=75e-6,emin=None):
    tau=T/ME; b2=beta2(T)
    xlo=I/T; xhi=min(0.5,0.5)
    x=np.linspace(xlo,xhi,200001)[1:-1]
    f=x*Fx(x,tau)
    val=np.trapezoid(f,x)
    return NE_w*2*np.pi*RE**2*ME/b2*val
for T in [0.1,0.5,1.0,2.0]:
    print(T, Scol_moller(T), xs.Scol(T))
