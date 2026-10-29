
# Check KN energy-fraction pdf normalisation: dsig/deps = pi re^2/k^2 * (1/eps+eps-sin^2)
from scipy.integrate import quad
def shape(eps,k):
    c=1-(1/eps-1)/k
    s2=1-c*c
    return 1/eps+eps-s2
def sig_from_shape(E):
    k=E/me; emin=1/(1+2*k)
    val,_=quad(lambda e: shape(e,k),emin,1,limit=200)
    return np.pi*re**2/k**2*val
print(sig_from_shape(2.0), sigKN(2.0))
