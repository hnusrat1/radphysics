# Let me debug the primary fluence normalization
import numpy as np
import math

# Quick check: count how many primary photons cross the cylinder
# Expected: N_src * exp(-mu*10) * (pi*1^2)/(11*11)
# = N_src * 0.607 * 0.026 = 0.0158 * N_src

# From MC with 50000 photons: primary track = 265.75 cm
# Each crossing adds 0.2 cm → 1329 crossings
# Fraction = 1329/50000 = 0.0266
# Expected: exp(-mu*10) * pi/121 = 0.607 * 0.02597 = 0.0158
# MC gives 0.0266, factor 1.68 higher

# So the issue is the number of photons crossing the cylinder is too high.
# This could be because:
# 1. The beam divergence is not correctly modeled (field at 10 cm is larger than I think?)
# 2. The cylinder radius is effectively larger
# 3. Some primary photons are tracked through the cylinder multiple times

# Let me check: with cylinder radius 1 cm at z=10, the beam at z=10 has half-width 5.5.
# Fraction of beam area covered by cylinder: pi/(11*11) = 0.026
# With attenuation: 0.607 * 0.026 = 0.0158

# Hmm, in the MC maybe the effective field is different. Let me check the beam divergence.
# At the surface (z=0), the field half-width is exactly 5 cm.
# The photon direction toward (x_t, y_t, 0) has components (x_t/|u|, y_t/|u|, SSD/|u|)
# At z=10, the photon position is: x = x_t * (SSD+10)/SSD 
# Because: position = u*t + start, and we need z = 10
# u_z = SSD/sqrt(x_t^2+y_t^2+SSD^2)
# t = 10/u_z, x = x_t * 10/u_z / |u| ... 

# Actually, let me compute directly: from source (0,0,-100) in direction u = (x_t, y_t, 100)/|u|
# At z=10: t*(u_z) = 110, so t = 110/u_z
# x = t*u_x = 110*x_t/|u| / (100/|u|) = 110*x_t/100 = 1.1*x_t
# So at z=10, the photon is at x = 1.1*x_t. For x_t = 5, x = 5.5 ✓

# But wait - the photon is tracked from the surface (z=0), not from the source!
# In the MC, I start the photon at the surface:
#   p_surf = pos + t_surf * u where pos = (0,0,-100)
#   t_surf = 100/uz, so p_surf = (x_t, y_t, 0)  (since u_x/u_z = x_t/100, x = 100*x_t/100 = x_t ✓)
# Then I call transport_photon(p_surf, u, E, ...)
# In transport_photon, the photon starts at p_surf = (x_t, y_t, 0) and travels in direction u.
# At z=10, x = x_t + 10*u_x/u_z = x_t + 10*x_t/100 = 1.1*x_t ✓

# So the beam is correct. Number of primary photons should be as expected.

# Wait, let me re-check the track length. When a primary photon crosses the cylinder:
# The photon travel direction is u = (x_t, y_t, 100)/|u|.
# For the trajectory through the cylinder: the photon enters through the cylinder top (z=10.1)
# and exits through the bottom (z=9.9) since the direction is nearly along z.
# Track length = 0.2/uz where uz ≈ 0.9997. So track ≈ 0.20006 cm.

# Hmm, let me check if my ray_cylinder_track function is correct.

# Let me test: photon at (0.5, 0.5, 9.9) heading toward (0.5, 0.5, 10.1)
p = np.array([0.5, 0.5, 9.9])
u = np.array([0.0, 0.0, 1.0])
# Should give 0.2
# From my function, it should.

# Let me instead check by counting photons:
# For a uniform distribution of source photons across the field, 
# the number that pass through the cylinder should be N_src * 0.026 * 0.607

# Let me run a simple geometric check: without attenuation (mu=0),
# count photons crossing the cylinder.
# Expected: N * pi/(11^2) = N * 0.026

print("Checking photon crossing fraction...")
np.random.seed(1)
N_test = 100000
count_cyl = 0
cyl_radius = 1.0
cyl_z_min = 9.9
cyl_z_max = 10.1
for i in range(N_test):
    x_t = (np.random.rand() - 0.5) * 10
    y_t = (np.random.rand() - 0.5) * 10
    u = np.array([x_t, y_t, 100.0])
    u = u / np.linalg.norm(u)
    # Position at z=10 (center of cylinder)
    z = 10.0
    t = z / u[2]
    x = t * u[0]
    y = t * u[1]
    if x*x + y*y <= cyl_radius*cyl_radius:
        # Check if the full path through the cylinder is within the field
        count_cyl += 1

frac = count_cyl / N_test
print(f"Fraction crossing cylinder at z=10: {frac:.5f}")
print(f"Expected (pi/121): {np.pi/121:.5f}")
print(f"Expected with attenuation (0.607*pi/121): {0.607*np.pi/121:.5f}")

# The discrepancy might be because the field at z=10 is 11x11, and the cylinder
# fraction should be pi/(11*11) = 0.026. My MC gives 0.0266 which is close but
# slightly higher (maybe edge effects).

# So the count is right, but the TRACK LENGTH might be wrong.

# Let me check my ray_cylinder_track function on the actual photon trajectory.
# Photon from source at (0.5, 0.5, -100) direction u=(0.5,0.5,100)/norm
u_test = np.array([0.5, 0.5, 100.0])
u_test = u_test / np.linalg.norm(u_test)
p_test = np.array([0.5, 0.5, 0.0])  # at surface

# Track length through cylinder at z=10
L = ray_cylinder_track(p_test, u_test)
print(f"\nTrack length for central field photon: {L:.6f} (expect ~0.2)")
# The photon goes from (0.5,0.5,0) in direction u_test
# At z=10, position = (0.5 + 10*u_test[0]/u_test[2], ...) 
xz = 10 * u_test[0]/u_test[2]
print(f"x at z=10 for photon at x_t=0.5: {xz:.6f} (expect 0.55)")
# Hmm, the photon passing through the cylinder at x_t=0.5 is at x=0.55 at z=10.
# Since it's within the cylinder, it's counted.

# The track through the cylinder: as the photon travels, it enters z=10.1 
# at one point and exits z=9.9. Track = 0.2/u_test[2] ≈ 0.2/0.99995 ≈ 0.20001

# So the track should be right. Let me check if the primary fluence 
# comes out right when accounting for the actual mu.

# Let me recompute the expected value with the actual mu from my model:
mu_val = mu_total(2.0)  # should be ~0.05
prim_expected_per_src = np.exp(-mu_val * 10) * np.pi / (11*11)  # this is the track-based fluence
print(f"\nmu(2 MeV) = {mu_val:.5f}")
print(f"Expected primary fluence per source photon: {prim_expected_per_src:.6f}")
print(f"MC value: {265.75/cyl_volume/50000:.6f}")