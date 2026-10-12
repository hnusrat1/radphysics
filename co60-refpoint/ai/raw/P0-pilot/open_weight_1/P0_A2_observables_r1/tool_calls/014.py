import numpy as np
from scipy.interpolate import interp1d

# Load and interpolate
xcom_data = np.loadtxt('/tmp/data/xcom_water.txt')
xcom_en = xcom_data[:, 0]
xcom_inc = xcom_data[:, 2]
xcom_tot = xcom_data[:, 6]

estar_data = np.loadtxt('/tmp/data/estar_water.txt')
estar_en = estar_data[:, 0]
estar_coll = estar_data[:, 1]
estar_rad = estar_data[:, 2]
estar_csda = estar_data[:, 3]
estar_rad_yield = estar_data[:, 4]

f_inc = interp1d(xcom_en, xcom_inc, kind='linear', bounds_error=False, fill_value=0.0)
f_tot = interp1d(xcom_en, xcom_tot, kind='linear', bounds_error=False, fill_value=0.0)
f_coll = interp1d(estar_en, estar_coll, kind='linear', bounds_error=False, fill_value=0.0)
f_rad = interp1d(estar_en, estar_rad, kind='linear', bounds_error=False, fill_value=0.0)
f_csda = interp1d(estar_en, estar_csda, kind='linear', bounds_error=False, fill_value=0.0)
f_rad_yield = interp1d(estar_en, estar_rad_yield, kind='linear', bounds_error=False, fill_value=0.0)

mec2 = 0.511
rho = 1.0
E0 = 2.0

# Geometry
SSD = 100.0
depth_POI = 10.0
scoring_z_min = 9.9
scoring_z_max = 10.1
scoring_r = 1.0
scoring_vol = np.pi * scoring_r**2 * 0.2

# Source at (0, 0, -SSD)
# Phantom from z=0 to z=30, x in [-15,15], y in [-15,15]

# For numerical integration, use cylindrical coordinates around the beam axis
# Using symmetry (the beam and scoring are on the central axis)

# Let's set up a fine grid for the numerical integration
# We'll use cylindrical coordinates (r, z) with azimuthal symmetry

# Number of grid points
nz = 200  # along depth
nr = 100  # radial

z_edges = np.linspace(0, 30, nz+1)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
dz = 30.0 / nz

r_edges = np.linspace(0, 15, nr+1)
r_centers = (r_edges[:-1] + r_edges[1:]) / 2
dr = 15.0 / nr

# For each voxel, compute the contribution to scattered fluence at scoring cylinder
# First, let's compute the primary photon flux at each voxel
# Primary photon travels from source (0,0,-SSD) to voxel at (r, phi, z)
# Distance from source to voxel:
# d_source_voxel = sqrt(r^2 + (z + SSD)^2)
# The primary fluence at the voxel (without attenuation) follows 1/d^2
# With attenuation: exp(-mu * path_in_water)

# The primary photon field is confined to the pyramid.
# At depth z, the field half-width is: field_size/2 * (z + SSD) / SSD
# field half-width at surface = 5 cm, at SSD=100
# So at depth z, half-width = 5 * (z + 100) / 100

def field_half_width(z):
    return 5.0 * (z + SSD) / SSD

# Primary photon fluence at voxel (r, z) relative to source strength
# Source emits uniformly into the pyramid
# Total number of photons in the beam: N0
# The fluence at a point in the field (without attenuation) is:
# N0 / (field_area_at_distance)
# field_area_at_distance(z) = (2*field_half_width(z))^2 = (10 * (z+SSD)/SSD)^2 = 100 * (z+SSD)^2 / SSD^2
# So fluence_without_attenuation = N0 * SSD^2 / (100 * (z+SSD)^2)
# Phi0 (fluence at 110 cm without phantom) = N0 * SSD^2 / (100 * 110^2)
# So the relative primary fluence (normalized to Phi0) at (r,z) without attenuation is:
# Phi0_rel = (110^2) / (z+SSD)^2

# With attenuation: Phi0_rel * exp(-mu_tot * path_in_water)
# Path in water from surface to depth z along the primary photon direction:
# For a point at (r, z), the primary photon enters the phantom at (r * SSD/(z+SSD), 0)
# Actually, it's easier: the path length in water is z / cos(alpha) where alpha is the angle
# from the central axis to the point
# cos(alpha) = (z+SSD) / sqrt(r^2 + (z+SSD)^2)
# Path in water = z / cos(alpha) = z * sqrt(r^2 + (z+SSD)^2) / (z+SSD)

# But for voxels near the axis (r small), path ≈ z

# Let's compute the scatter contribution
print("Setting up integration grid...")
print(f"nz={nz}, nr={nr}")
print(f"Number of voxels: {nz*nr}")

# We'll compute the scattered photon fluence at the scoring cylinder
# The scoring cylinder: centered at z=10, radius 1, height 0.2
# For a scattered photon originating at voxel (r_v, z_v), we need to know:
# 1. Scattering angle (between incident direction and scattered direction to scoring volume)
# 2. Energy of scattered photon (from Compton formula)
# 3. Attenuation of scattered photon from voxel to scoring volume

# The incident direction at the voxel: from source to voxel
# The scattered direction: from voxel to scoring cylinder center (0, z=10)

# Let's compute for a grid of voxels
# Since the scoring cylinder is small and on-axis, we can approximate the scatter contribution
# by evaluating at the cylinder center

print("Computing scatter contribution...")

# Pre-compute values
# Source position: (0, 0, -SSD)
# Scoring center: (0, 0, 10)

L1_scattered_fluence = 0.0  # Will accumulate scattered fluence / Phi0
scattered_photon_spectrum = np.zeros(28)  # Fluence per bin / Phi0
scattered_angular = np.zeros(10)  # cos theta bins

# Electron initial spectrum
electron_initial_spectrum = np.zeros(25)
electron_initial_spectrum_primary = np.zeros(25)
electron_initial_direction = np.zeros(10)

# Interaction counts
n_interactions_total = 0
n_interactions_primary = 0
n_interactions_scattered = 0
n_compton = 0
n_photo = 0
n_pair = 0
n_rayleigh = 0
n_compton_pri = 0
n_photo_pri = 0
n_pair_pri = 0
n_rayleigh_pri = 0
n_compton_scat = 0
n_photo_scat = 0
n_pair_scat = 0
n_rayleigh_scat = 0

# Energy transferred to charged particles
E_transfer_compton = 0.0
E_transfer_photo = 0.0
E_transfer_pair = 0.0
E_transfer_rayleigh = 0.0

# For scattered fluence, we'll only consider first Compton scatter
# Multiple scatter and other processes contribute less

# Let me first compute the direct scattered contribution more carefully
# using a subset of the grid to be faster

# Actually, let me use a coarser grid first to test
nz_test = 50
nr_test = 25

z_centers_test = np.linspace(0.3, 29.7, nz_test)
r_centers_test = np.linspace(0.15, 14.85, nr_test)
dz_test = 30.0 / nz_test
dr_test = 15.0 / nr_test

print("Testing with coarse grid...")

# Photon bins
photon_bins = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4])

# Electron bins
electron_bins = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.4])

# cos theta bins
cos_theta_bins = np.array([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])