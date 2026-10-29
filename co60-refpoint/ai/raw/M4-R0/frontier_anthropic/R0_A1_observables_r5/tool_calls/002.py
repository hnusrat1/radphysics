
def check(E):
    k=E/me
    def g(c):
        eps=1/(1+k*(1-c))
        dsig=0.5*re**2*eps**2*(eps+1/eps-(1-c**2))
        return dsig*2*np.pi
    s,_=integrate.quad(g,-1,1)
    return s, kn_sigma(E)
print(check(1.25), check(0.1))
