import numpy as np
import math

# Quick geometry test

def sample_beam_direction():
    """Sample a photon direction uniformly within the beam pyramid"""
    x0 = np.random.uniform(-5, 5)
    y0 = np.random.uniform(-5, 5)
    dx = x0
    dy = y0
    dz = 100.0  # distance from source to surface
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    return np.array([dx/norm, dy/norm, dz/norm])

def point_in_phantom(pos):
    x, y, z = pos
    return abs(x) <= 15 and abs(y) <= 15 and 0 <= z <= 30

def distance_to_phantom_boundary(pos, dir_vec):
    """Distance along direction from pos to phantom boundary."""
    x, y, z = pos
    dx, dy, dz = dir_vec
    
    t_min = float('inf')
    
    # x boundaries
    if dx > 1e-12:
        t = (15 - x) / dx
        if t > 0:
            t_min = min(t_min, t)
    elif dx < -1e-12:
        t = (-15 - x) / dx
        if t > 0:
            t_min = min(t_min, t)
    
    # y boundaries
    if dy > 1e-12:
        t = (15 - y) / dy
        if t > 0:
            t_min = min(t_min, t)
    elif dy < -1e-12:
        t = (-15 - y) / dy
        if t > 0:
            t_min = min(t_min, t)
    
    # z boundaries
    if dz > 1e-12:
        t = (30 - z) / dz
        if t > 0:
            t_min = min(t_min, t)
    elif dz < -1e-12:
        t = (0 - z) / dz
        if t > 0:
            t_min = min(t_min, t)
    
    return t_min

# Test: source at (-100), sample direction, transport to phantom surface
source = np.array([0.0, 0.0, -100.0])
for i in range(10):
    dir_vec = sample_beam_direction()
    print(f"Direction: {dir_vec}, z-component: {dir_vec[2]:.6f}")
    
    # Check if moving toward phantom
    if dir_vec[2] <= 0:
        print("  Not moving toward phantom!")
        continue
    
    # Distance to surface z=0
    t_surf = (0 - source[2]) / dir_vec[2]
    pos_at_surface = source + dir_vec * t_surf
    print(f"  At surface: ({pos_at_surface[0]:.4f}, {pos_at_surface[1]:.4f}, {pos_at_surface[2]:.4f})")
    print(f"  In phantom: {point_in_phantom(pos_at_surface)}")
    
    # Distance to phantom boundary at this point
    d_exit = distance_to_phantom_boundary(pos_at_surface, dir_vec)
    print(f"  Distance to exit: {d_exit:.4f}")
    
    # At 10 cm depth
    pos_10 = source + dir_vec * (110.0 / dir_vec[2])  # z goes from -100 to 10
    print(f"  At 10 cm depth: ({pos_10[0]:.4f}, {pos_10[1]:.4f}, {pos_10[2]:.4f})")
    print(f"  Radius: {math.sqrt(pos_10[0]**2 + pos_10[1]**2):.4f}")
    print(f"  In scoring volume: {pos_10[2] >= 9.9 and pos_10[2] <= 10.1 and math.sqrt(pos_10[0]**2 + pos_10[1]**2) <= 1.0}")