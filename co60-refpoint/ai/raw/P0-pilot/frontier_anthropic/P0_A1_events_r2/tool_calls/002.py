
import re
src = open('/tmp/work/_call_001.py').read() if False else None
def sgn(x): return 1.0 if x>=0 else -1.0
def rotate(d, cost, phi):
    ux,uy,uz = d
    sint = math.sqrt(max(0.0,1-cost*cost))
    if abs(uz) > 0.99999:
        nx = sint*math.cos(phi); ny = sint*math.sin(phi); nz = cost*sgn(uz)
    else:
        temp = math.sqrt(1-uz*uz)
        nx = sint*(ux*uz*math.cos(phi) - uy*math.sin(phi))/temp + ux*cost
        ny = sint*(uy*uz*math.cos(phi) + ux*math.sin(phi))/temp + uy*cost
        nz = -sint*math.cos(phi)*temp + uz*cost
    n = math.sqrt(nx*nx+ny*ny+nz*nz)
    return (nx/n, ny/n, nz/n)

def mu_pair(E):
    if E < 2*M_E: return 0.0
    return 0.0006*min(1.0,(E-2*M_E)/(2.0-2*M_E))

events = []
N = 300000
t0 = time.time()
for i in range(N):
    p,d = sample_entry_dir()
    track_photon(2.0, p, d, events, None)
print("histories", N, "events in volume", len(events), "time", round(time.time()-t0,1))
print(Counter(e['process'] for e in events))
