#!/usr/bin/env python3
"""
Enhanced Warp-based Bedsheet Simulation with Aerodynamic Flow
High-performance cloth simulation using NVIDIA Warp with realistic air flow effects
"""

import warp as wp
import numpy as np
import os
import json
from pathlib import Path
import argparse
import random


# Initialize Warp
wp.init()


@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3
):
    """Integrate particle positions and velocities"""
    tid = wp.tid()
    
    # Apply gravity and forces
    v_new = v[tid] + (f[tid] * inv_mass[tid] + gravity) * dt
    
    # Update position
    x[tid] = x[tid] + v_new * dt
    v[tid] = v_new


@wp.kernel
def apply_distance_constraints(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    constraint_count: wp.array(dtype=int),
    constraint_indices: wp.array(dtype=wp.vec2i),
    constraint_rest_lengths: wp.array(dtype=float),
    stiffness: float,
    iterations: int
):
    """Apply distance constraints for cloth structure"""
    tid = wp.tid()
    
    if tid >= constraint_count[0]:
        return
    
    # Get constraint indices
    i = constraint_indices[tid][0]
    j = constraint_indices[tid][1]
    
    # Get current positions
    p1 = x[i]
    p2 = x[j]
    
    # Calculate current distance
    diff = p1 - p2
    current_length = wp.length(diff)
    rest_length = constraint_rest_lengths[tid]
    
    if current_length > 0.0:
        # Calculate correction
        correction = (current_length - rest_length) * stiffness
        direction = diff / current_length
        
        # Apply correction based on inverse mass
        w1 = inv_mass[i]
        w2 = inv_mass[j]
        w_sum = w1 + w2
        
        if w_sum > 0.0:
            correction1 = -correction * w1 / w_sum
            correction2 = correction * w2 / w_sum
            
            x[i] = x[i] + direction * correction1
            x[j] = x[j] + direction * correction2


@wp.kernel
def apply_bending_constraints(
    x: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    bend_count: wp.array(dtype=int),
    bend_indices: wp.array(dtype=wp.vec4i),
    bend_rest_angles: wp.array(dtype=float),
    stiffness: float
):
    """Apply bending constraints for cloth stiffness"""
    tid = wp.tid()
    
    if tid >= bend_count[0]:
        return
    
    # Get bending constraint indices (4 vertices forming two adjacent triangles)
    i = bend_indices[tid][0]
    j = bend_indices[tid][1]
    k = bend_indices[tid][2]
    l = bend_indices[tid][3]
    
    # Get positions
    p1 = x[i]
    p2 = x[j]
    p3 = x[k]
    p4 = x[l]
    
    # Calculate dihedral angle between triangles
    n1 = wp.cross(p2 - p1, p3 - p1)
    n2 = wp.cross(p2 - p1, p4 - p1)
    
    n1_len = wp.length(n1)
    n2_len = wp.length(n2)
    
    if n1_len > 0.0 and n2_len > 0.0:
        n1 = n1 / n1_len
        n2 = n2 / n2_len
        
        # Calculate angle between normals
        cos_angle = wp.dot(n1, n2)
        cos_angle = wp.clamp(cos_angle, -1.0, 1.0)
        
        # Apply bending constraint
        rest_angle = bend_rest_angles[tid]
        angle_diff = wp.acos(cos_angle) - rest_angle
        
        if abs(angle_diff) > 0.01:  # Threshold to avoid numerical issues
            # Calculate gradient and apply correction
            correction = angle_diff * stiffness * 0.1
            
            # Apply small corrections to maintain cloth structure
            x[i] = x[i] + n1 * correction * 0.1
            x[j] = x[j] + n2 * correction * 0.1


@wp.kernel
def apply_aerodynamic_forces(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    wind_velocity: wp.vec3,
    air_density: float,
    drag_coefficient: float,
    lift_coefficient: float,
    wind_noise_scale: float,
    wind_noise_strength: float,
    wind_turbulence: float,
    time: float,
    nx: int,
    ny: int
):
    """Apply aerodynamic forces with enhanced wind noise and turbulence"""
    tid = wp.tid()
    
    if tid >= x.shape[0]:
        return
    
    # Get particle position and velocity
    pos = x[tid]
    vel = v[tid]
    
    # Ultra-chaotic wind noise calculation with multiple frequency layers
    # Primary noise layer
    noise_x = wp.sin(pos.x * wind_noise_scale + time * 0.5) * wp.cos(pos.y * wind_noise_scale * 0.7 + time * 0.3)
    noise_y = wp.cos(pos.y * wind_noise_scale + time * 0.4) * wp.sin(pos.x * wind_noise_scale * 0.8 + time * 0.6)
    noise_z = wp.sin(pos.z * wind_noise_scale * 0.5 + time * 0.2) * wp.cos(pos.x * wind_noise_scale * 0.3 + pos.y * wind_noise_scale * 0.4)
    
    # High frequency noise for fine-scale chaos
    fine_noise_x = wp.sin(pos.x * wind_noise_scale * 3.0 + time * 1.2) * 0.5
    fine_noise_y = wp.cos(pos.y * wind_noise_scale * 2.5 + time * 0.8) * 0.5
    fine_noise_z = wp.sin(pos.z * wind_noise_scale * 4.0 + time * 1.5) * 0.3
    
    # Ultra-high frequency noise for maximum chaos
    chaos_noise_x = wp.sin(pos.x * wind_noise_scale * 6.0 + time * 2.5) * 0.4
    chaos_noise_y = wp.cos(pos.y * wind_noise_scale * 5.5 + time * 2.0) * 0.4
    chaos_noise_z = wp.sin(pos.z * wind_noise_scale * 8.0 + time * 3.0) * 0.2
    
    # Random chaotic bursts
    burst_factor = wp.sin(time * 5.0 + pos.x * 10.0) * wp.cos(time * 7.0 + pos.y * 8.0)
    burst_noise = wp.vec3(
        burst_factor * 0.6,
        burst_factor * 0.4,
        burst_factor * 0.2
    )
    
    # Combine all noise components for maximum chaos
    turbulence = wp.vec3(
        (noise_x + fine_noise_x + chaos_noise_x + burst_noise.x) * wind_noise_strength,
        (noise_y + fine_noise_y + chaos_noise_y + burst_noise.y) * wind_noise_strength,
        (noise_z + fine_noise_z + chaos_noise_z + burst_noise.z) * wind_noise_strength * 0.7  # More vertical chaos
    )
    
    # Enhanced time-varying wind gusts with multiple frequencies
    gust_strength = (wp.sin(time * 0.8) * wp.cos(time * 1.3) + 
                    wp.sin(time * 2.1) * wp.cos(time * 1.7) * 0.5 +
                    wp.sin(time * 4.2) * wp.cos(time * 3.1) * 0.3) * wind_turbulence
    gust_direction = wp.vec3(
        wp.sin(time * 0.5) + wp.sin(time * 1.8) * 0.3,
        wp.cos(time * 0.7) + wp.cos(time * 2.2) * 0.3,
        wp.sin(time * 1.2) * 0.4  # Add more vertical gust component
    )
    gust_direction = wp.normalize(gust_direction)
    
    # Apply turbulence and gusts to wind
    local_wind = wind_velocity + turbulence + gust_direction * gust_strength
    local_rel_vel = local_wind - vel
    local_rel_vel_mag = wp.length(local_rel_vel)
    
    if local_rel_vel_mag > 0.001:
        local_rel_vel_dir = local_rel_vel / local_rel_vel_mag
        
        # Calculate aerodynamic force magnitude
        # F = 0.5 * rho * v^2 * Cd * A
        particle_area = 0.01  # m^2 (approximate area per particle)
        force_mag = 0.5 * air_density * local_rel_vel_mag * local_rel_vel_mag * drag_coefficient * particle_area
        
        # Apply drag force (opposite to relative velocity)
        drag_force = -local_rel_vel_dir * force_mag
        
        # Enhanced lift force calculation
        lift_force = wp.vec3(0.0, 0.0, 0.0)
        
        if local_rel_vel_mag > 0.5:  # Apply lift for moderate velocities
            lift_magnitude = 0.5 * air_density * local_rel_vel_mag * local_rel_vel_mag * lift_coefficient * particle_area
            
            # More sophisticated lift direction calculation
            # Lift perpendicular to velocity and influenced by local surface normal
            lift_direction = wp.vec3(-local_rel_vel_dir.y, local_rel_vel_dir.x, 0.0)
            lift_direction = wp.normalize(lift_direction)
            
            # Add upward component based on velocity magnitude
            upward_component = wp.vec3(0.0, 0.0, 1.0) * wp.min(local_rel_vel_mag * 0.1, 0.5)
            lift_direction = lift_direction + upward_component
            lift_direction = wp.normalize(lift_direction)
            
            lift_force = lift_direction * lift_magnitude * 0.4  # Increased lift strength
        
        # Apply forces
        total_force = drag_force + lift_force
        f[tid] = f[tid] + total_force


@wp.kernel
def apply_side_wind(
    x: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    side_wind_strength: float,
    side_wind_direction: float,
    time: float
):
    """Apply strong side wind forces that blow from all sides of the bedsheet"""
    tid = wp.tid()
    
    if tid >= x.shape[0]:
        return
    
    pos = x[tid]
    
    # Calculate rotating side wind direction with faster rotation
    current_direction = side_wind_direction + time * 1.5  # Faster rotation for more chaos
    
    # Strong side wind comes from all sides (X and Y directions)
    side_wind_force = wp.vec3(
        wp.cos(current_direction) * side_wind_strength,
        wp.sin(current_direction) * side_wind_strength,
        wp.sin(current_direction * 0.7) * side_wind_strength * 0.3  # Add some vertical component
    )
    
    # Add multiple layers of position-based variation for chaotic side wind
    pos_factor_x = wp.sin(pos.x * 1.0 + time * 0.8) * wp.cos(pos.y * 0.6 + time * 0.5)
    pos_factor_y = wp.cos(pos.y * 1.2 + time * 0.6) * wp.sin(pos.x * 0.8 + time * 0.7)
    pos_factor_z = wp.sin(pos.z * 0.5 + time * 0.4) * wp.cos(pos.x * 0.3 + pos.y * 0.4)
    
    # Combine position factors for more chaotic variation
    pos_factor = (pos_factor_x + pos_factor_y + pos_factor_z) * 0.3
    side_wind_force = side_wind_force * (0.7 + pos_factor * 0.6)  # More variation
    
    f[tid] = f[tid] + side_wind_force


@wp.kernel
def apply_omnidirectional_wind(
    x: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    wind_strength: float,
    time: float
):
    """Apply strong wind forces from all directions simultaneously"""
    tid = wp.tid()
    
    if tid >= x.shape[0]:
        return
    
    pos = x[tid]
    
    # Create wind forces from all 8 cardinal and diagonal directions
    # North
    north_force = wp.vec3(0.0, wind_strength, 0.0)
    # South  
    south_force = wp.vec3(0.0, -wind_strength, 0.0)
    # East
    east_force = wp.vec3(wind_strength, 0.0, 0.0)
    # West
    west_force = wp.vec3(-wind_strength, 0.0, 0.0)
    # Northeast
    ne_force = wp.vec3(wind_strength * 0.7, wind_strength * 0.7, 0.0)
    # Northwest
    nw_force = wp.vec3(-wind_strength * 0.7, wind_strength * 0.7, 0.0)
    # Southeast
    se_force = wp.vec3(wind_strength * 0.7, -wind_strength * 0.7, 0.0)
    # Southwest
    sw_force = wp.vec3(-wind_strength * 0.7, -wind_strength * 0.7, 0.0)
    
    # Add time-varying intensity for each direction
    time_factor = wp.sin(time * 2.0) * 0.5 + 0.5  # Oscillates between 0 and 1
    
    # Combine all directional forces with time variation
    total_force = (north_force + south_force + east_force + west_force + 
                   ne_force + nw_force + se_force + sw_force) * time_factor * 0.125
    
    # Add chaotic noise to the combined force
    noise_x = wp.sin(pos.x * 2.0 + time * 1.5) * wp.cos(pos.y * 1.8 + time * 1.2) * wind_strength * 0.3
    noise_y = wp.cos(pos.y * 2.2 + time * 1.8) * wp.sin(pos.x * 1.6 + time * 1.4) * wind_strength * 0.3
    noise_z = wp.sin(pos.z * 1.0 + time * 0.8) * wind_strength * 0.2
    
    total_force = total_force + wp.vec3(noise_x, noise_y, noise_z)
    
    f[tid] = f[tid] + total_force


@wp.kernel
def apply_multiple_wind_sources(
    x: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    wind_centers: wp.array(dtype=wp.vec3),
    wind_strengths: wp.array(dtype=float),
    wind_radii: wp.array(dtype=float),
    wind_directions: wp.array(dtype=float),
    wind_rotation_speeds: wp.array(dtype=float),
    wind_noise_scale: float,
    wind_noise_strength: float,
    wind_turbulence: float,
    time: float,
    num_sources: int
):
    """Apply multiple rotating wind sources for maximum chaos"""
    tid = wp.tid()
    
    if tid >= x.shape[0]:
        return
    
    pos = x[tid]
    total_wind_force = wp.vec3(0.0, 0.0, 0.0)
    
    # Apply each wind source
    for i in range(num_sources):
        if i >= wind_centers.shape[0]:
            break
            
        wind_center = wind_centers[i]
        wind_strength = wind_strengths[i]
        wind_radius = wind_radii[i]
        wind_direction = wind_directions[i]
        rotation_speed = wind_rotation_speeds[i]
        
        # Calculate distance from wind center
        dist = wp.length(pos - wind_center)
        
        if dist < wind_radius:
            # Calculate wind strength based on distance
            strength_factor = 1.0 - (dist / wind_radius)
            strength_factor = strength_factor * strength_factor
            
            # Rotating wind direction
            current_direction = wind_direction + time * rotation_speed
            
            # Calculate wind direction with rotation
            wind_dir = wp.vec3(
                wp.cos(current_direction),
                wp.sin(current_direction),
                wp.sin(current_direction * 0.5) * 0.3  # Some vertical component
            )
            
            # Add extreme noise and turbulence
            noise_x = (wp.sin(pos.x * wind_noise_scale + time * 0.8) * 0.4 +
                      wp.sin(pos.x * wind_noise_scale * 3.0 + time * 1.5) * 0.3 +
                      wp.sin(pos.x * wind_noise_scale * 6.0 + time * 2.8) * 0.2) * wind_noise_strength
            noise_y = (wp.cos(pos.y * wind_noise_scale + time * 0.6) * 0.4 +
                      wp.cos(pos.y * wind_noise_scale * 2.5 + time * 1.2) * 0.3 +
                      wp.cos(pos.y * wind_noise_scale * 5.0 + time * 2.1) * 0.2) * wind_noise_strength
            noise_z = (wp.sin(pos.z * wind_noise_scale * 0.8 + time * 0.4) * 0.3 +
                      wp.sin(pos.z * wind_noise_scale * 2.0 + time * 0.9) * 0.2) * wind_noise_strength
            
            # Combine wind direction with noise
            final_direction = wind_dir + wp.vec3(noise_x, noise_y, noise_z)
            final_direction = wp.normalize(final_direction)
            
            # Apply wind force
            wind_force = final_direction * wind_strength * strength_factor
            total_wind_force = total_wind_force + wind_force
    
    f[tid] = f[tid] + total_wind_force


@wp.kernel
def apply_wind_field(
    x: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    wind_center: wp.vec3,
    wind_strength: float,
    wind_radius: float,
    wind_noise_scale: float,
    wind_noise_strength: float,
    wind_turbulence: float,
    time: float
):
    """Apply enhanced localized wind field with noise and turbulence"""
    tid = wp.tid()
    
    if tid >= x.shape[0]:
        return
    
    pos = x[tid]
    
    # Calculate distance from wind center
    dist = wp.length(pos - wind_center)
    
    if dist < wind_radius:
        # Calculate wind strength based on distance (stronger at center)
        strength_factor = 1.0 - (dist / wind_radius)
        strength_factor = strength_factor * strength_factor  # Quadratic falloff
        
        # Ultra-chaotic time-based variation with many frequencies
        time_variation = (wp.sin(time * 2.0) * 0.2 + 
                         wp.sin(time * 0.7) * 0.3 + 
                         wp.sin(time * 3.2) * 0.2 +
                         wp.sin(time * 5.5) * 0.15 +
                         wp.sin(time * 8.1) * 0.1 +
                         wp.sin(time * 12.3) * 0.05) * 0.5 + 0.5
        
        # Enhanced wind gusts with chaotic frequencies
        gust_variation = (wp.sin(time * 1.5) * wp.cos(time * 2.3) * 0.4 +
                         wp.sin(time * 4.2) * wp.cos(time * 3.7) * 0.3 +
                         wp.sin(time * 7.1) * wp.cos(time * 5.9) * 0.2 +
                         wp.sin(time * 11.3) * wp.cos(time * 8.4) * 0.1) + 1.0
        
        # Calculate ultra-chaotic wind direction with multiple noise layers
        base_direction = wp.vec3(
            wp.sin(time * 0.5) + wp.sin(time * 1.3) * 0.3 + wp.sin(time * 2.7) * 0.2,
            wp.cos(time * 0.3) + wp.cos(time * 1.1) * 0.3 + wp.cos(time * 2.4) * 0.2,
            wp.sin(time * 0.8) * 0.4 + wp.sin(time * 1.9) * 0.2  # More vertical chaos
        )
        
        # Add multiple layers of position-based noise
        noise_x = (wp.sin(pos.x * wind_noise_scale + time * 0.6) * 0.3 +
                  wp.sin(pos.x * wind_noise_scale * 2.0 + time * 1.2) * 0.2 +
                  wp.sin(pos.x * wind_noise_scale * 4.0 + time * 2.1) * 0.1) * wind_noise_strength
        noise_y = (wp.cos(pos.y * wind_noise_scale + time * 0.4) * 0.3 +
                  wp.cos(pos.y * wind_noise_scale * 2.5 + time * 1.5) * 0.2 +
                  wp.cos(pos.y * wind_noise_scale * 5.0 + time * 2.8) * 0.1) * wind_noise_strength
        noise_z = (wp.sin(pos.z * wind_noise_scale * 0.5 + time * 0.3) * 0.2 +
                  wp.sin(pos.z * wind_noise_scale * 1.5 + time * 0.9) * 0.15 +
                  wp.sin(pos.z * wind_noise_scale * 3.0 + time * 1.7) * 0.1) * wind_noise_strength
        
        # Combine base direction with all noise layers
        wind_direction = base_direction + wp.vec3(noise_x, noise_y, noise_z)
        wind_direction = wp.normalize(wind_direction)
        
        # Enhanced turbulence effects with multiple frequencies
        turbulence_factor = (1.0 + 
                           wp.sin(time * 4.0 + pos.x * 2.0) * wind_turbulence * 0.4 +
                           wp.sin(time * 7.2 + pos.y * 3.1) * wind_turbulence * 0.3 +
                           wp.sin(time * 11.5 + pos.z * 1.8) * wind_turbulence * 0.2)
        
        # Apply enhanced wind force
        wind_force = wind_direction * wind_strength * strength_factor * time_variation * gust_variation * turbulence_factor
        f[tid] = f[tid] + wind_force


@wp.kernel
def apply_air_resistance(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    air_resistance_coefficient: float,
    linear_damping: float
):
    """Apply overall air resistance to all particles"""
    tid = wp.tid()
    
    if tid >= v.shape[0]:
        return
    
    # Get current velocity
    vel = v[tid]
    vel_mag = wp.length(vel)
    
    if vel_mag > 0.001:  # Avoid division by zero
        # Apply air resistance force (opposite to velocity)
        air_resistance_force = -vel * air_resistance_coefficient * vel_mag
        f[tid] = f[tid] + air_resistance_force
        
        # Apply linear damping to velocity
        v[tid] = vel * linear_damping


@wp.kernel
def clear_forces(
    f: wp.array(dtype=wp.vec3)
):
    """Clear force array"""
    tid = wp.tid()
    f[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    ground_height: float,
    thickness: float,
    restitution: float,
    friction: float
):
    """Apply ground collision with friction, accounting for bedsheet thickness"""
    tid = wp.tid()
    
    # Account for thickness - collision happens when bottom of bedsheet hits ground
    collision_height = ground_height + thickness * 0.5
    
    if x[tid][2] < collision_height:
        # Collision with ground - position at ground level plus half thickness
        x[tid] = wp.vec3(x[tid][0], x[tid][1], collision_height)
        
        # Apply restitution and friction
        v_new = v[tid]
        v_new = wp.vec3(v_new[0], v_new[1], -v_new[2] * restitution)
        
        # Apply friction
        v_horizontal = wp.vec3(v_new[0], v_new[1], 0.0)
        v_horizontal_len = wp.length(v_horizontal)
        
        if v_horizontal_len > 0.0:
            friction_force = friction * v_horizontal_len
            if friction_force > v_horizontal_len:
                v_new = wp.vec3(0.0, 0.0, v_new[2])
            else:
                v_horizontal = v_horizontal * (1.0 - friction)
                v_new = wp.vec3(v_horizontal[0], v_horizontal[1], v_new[2])
        
        v[tid] = v_new


@wp.kernel
def check_landing_status(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    ground_height: float,
    landing_threshold: float,
    velocity_threshold: float,
    landing_status: wp.array(dtype=int)
):
    """Check if bedsheet has landed and settled"""
    tid = wp.tid()
    
    if tid >= x.shape[0]:
        return
    
    # Check if particle is near ground and has low velocity
    height_above_ground = x[tid][2] - ground_height
    velocity_magnitude = wp.length(v[tid])
    
    if height_above_ground < landing_threshold and velocity_magnitude < velocity_threshold:
        # This particle has landed
        landing_status[tid] = 1
    else:
        landing_status[tid] = 0


class EnhancedWarpBedsheetSimulator:
    """Enhanced Warp-based bedsheet simulator with aerodynamic effects"""
    
    def __init__(self, width=2.0, height=1.0, resolution=64):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Calculate grid dimensions with higher resolution
        self.nx = int(resolution * width / max(width, height))
        self.ny = int(resolution * height / max(width, height))
        self.num_particles = self.nx * self.ny
        
        # Simulation parameters - soft but not stretchable
        self.dt = 0.02  # 50 FPS - faster timestep for speed
        self.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.damping = 0.95  # Less damping for more dynamic motion
        self.stiffness = 0.999  # Very high stiffness to prevent stretching (increased)
        self.bend_stiffness = 0.05  # Low bend stiffness for soft draping
        
        # Ground collision
        self.ground_height = 0.0
        self.restitution = 0.3
        self.friction = 0.8
        
        # Aerodynamic parameters
        self.air_density = 1.225  # kg/m^3 (sea level)
        self.drag_coefficient = 1.2  # Typical for cloth
        self.lift_coefficient = 0.3  # Lower for cloth
        
        # Air resistance parameters
        self.air_resistance_coefficient = 0.1  # Overall air resistance multiplier
        self.linear_damping = 0.95  # Linear velocity damping
        self.angular_damping = 0.98  # Angular velocity damping
        
        # Moderate wind parameters from all sides
        self.base_wind_velocity = wp.vec3(
            random.uniform(-4.0, 4.0),  # Moderate wind from all X directions
            random.uniform(-4.0, 4.0),  # Moderate wind from all Y directions
            random.uniform(-1.5, 1.5)   # Moderate vertical wind component
        )
        self.wind_strength = random.uniform(3.0, 6.0)  # Moderate wind strength
        self.wind_radius = random.uniform(2.5, 4.0)  # Moderate wind field size
        self.wind_center = wp.vec3(
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(1.5, 3.0)  # Moderate wind center height
        )
        
        # Moderate wind noise parameters for natural variation
        self.wind_noise_scale = random.uniform(1.0, 2.0)  # Moderate frequency noise
        self.wind_noise_strength = random.uniform(1.5, 3.0)  # Moderate noise amplitude
        self.wind_turbulence = random.uniform(0.8, 1.5)  # Moderate turbulence for natural motion
        
        # Moderate side wind parameters for lateral forces from all sides
        self.side_wind_strength = random.uniform(2.0, 4.0)  # Moderate side wind
        self.side_wind_direction = random.uniform(0, 2 * np.pi)  # Random initial direction
        self.wind_rotation_speed = random.uniform(0.5, 1.0)  # Moderate wind rotation
        
        # Moderate omnidirectional wind parameters for natural variation from all sides
        self.omnidirectional_wind_strength = random.uniform(3.0, 6.0)  # Moderate omnidirectional wind
        
        # Multiple moderate wind sources from all directions
        self.num_wind_sources = random.randint(2, 4)  # 2-4 wind sources for natural variation
        self.wind_sources = []
        for i in range(self.num_wind_sources):
            wind_source = {
                'center': wp.vec3(
                    random.uniform(-2.5, 2.5),  # Moderate coverage
                    random.uniform(-2.5, 2.5),
                    random.uniform(1.0, 3.0)  # Moderate wind source height
                ),
                'strength': random.uniform(2.0, 5.0),  # Moderate wind sources
                'radius': random.uniform(1.5, 3.0),  # Moderate wind fields
                'direction': random.uniform(0, 2 * np.pi),
                'rotation_speed': random.uniform(0.2, 0.8)  # Moderate rotation
            }
            self.wind_sources.append(wind_source)
        
        # Landing detection
        self.landing_threshold = 0.05  # 5cm above ground
        self.velocity_threshold = 0.1  # 0.1 m/s
        self.landing_percentage_threshold = 0.8  # 80% of particles must land
        
        # Bedsheet thickness
        self.thickness = 0.002  # 2mm thickness for realistic bedsheet
        
        # Initialize arrays
        self._setup_arrays()
        self._setup_constraints()
        self._setup_wind_sources()
        
        print(f"Initialized Enhanced Warp bedsheet simulator:")
        print(f"  Grid: {self.nx}x{self.ny} ({self.num_particles} particles)")
        print(f"  Size: {width}m x {height}m")
        print(f"  Resolution: {resolution}")
        print(f"  Thickness: {self.thickness}m")
        print(f"  Air resistance: {self.air_resistance_coefficient}")
        print(f"  Wind: {self.base_wind_velocity} (strength: {self.wind_strength})")
        print(f"  Wind center: {self.wind_center} (radius: {self.wind_radius})")
        print(f"  Wind noise scale: {self.wind_noise_scale}")
        print(f"  Wind noise strength: {self.wind_noise_strength}")
        print(f"  Wind turbulence: {self.wind_turbulence}")
        print(f"  Side wind strength: {self.side_wind_strength}")
        print(f"  Wind rotation speed: {self.wind_rotation_speed}")
        print(f"  Omnidirectional wind strength: {self.omnidirectional_wind_strength}")
        print(f"  Number of wind sources: {self.num_wind_sources}")
        for i, source in enumerate(self.wind_sources):
            print(f"    Wind source {i+1}: strength={source['strength']:.1f}, radius={source['radius']:.1f}")
    
    def _setup_arrays(self):
        """Setup Warp arrays for simulation"""
        # Particle arrays
        self.x = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.v = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.f = wp.zeros(self.num_particles, dtype=wp.vec3)
        self.inv_mass = wp.zeros(self.num_particles, dtype=float)
        
        # Landing detection array
        self.landing_status = wp.zeros(self.num_particles, dtype=int)
        
        # Initialize particle positions
        x_host = np.zeros((self.num_particles, 3), dtype=np.float32)
        inv_mass_host = np.ones(self.num_particles, dtype=np.float32)
        
        # Start bedsheet higher for more dramatic falling
        start_height = random.uniform(2.0, 3.0)
        
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                x_host[idx] = [
                    (i / (self.nx - 1) - 0.5) * self.width,
                    (j / (self.ny - 1) - 0.5) * self.height,
                    start_height
                ]
        
        # Pin corner particles (set inv_mass to 0) - but only initially
        # We'll release them after a few frames for more natural motion
        corner_indices = [0, self.ny-1, (self.nx-1)*self.ny, (self.nx-1)*self.ny + self.ny-1]
        for idx in corner_indices:
            inv_mass_host[idx] = 0.0
        
        # Copy to GPU
        self.x = wp.from_numpy(x_host, dtype=wp.vec3)
        self.inv_mass = wp.from_numpy(inv_mass_host, dtype=float)
        
        # Track when to release corners
        self.corner_release_frame = random.randint(30, 60)
        self.corners_released = False
    
    def _setup_constraints(self):
        """Setup distance and bending constraints"""
        # Distance constraints (springs between adjacent particles)
        constraints = []
        
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                
                # Horizontal springs
                if i < self.nx - 1:
                    right_idx = (i + 1) * self.ny + j
                    rest_length = self.width / (self.nx - 1)
                    constraints.append((idx, right_idx, rest_length))
                
                # Vertical springs
                if j < self.ny - 1:
                    up_idx = i * self.ny + (j + 1)
                    rest_length = self.height / (self.ny - 1)
                    constraints.append((idx, up_idx, rest_length))
                
                # Diagonal springs for stability
                if i < self.nx - 1 and j < self.ny - 1:
                    diag_idx = (i + 1) * self.ny + (j + 1)
                    rest_length = np.sqrt((self.width / (self.nx - 1))**2 + (self.height / (self.ny - 1))**2)
                    constraints.append((idx, diag_idx, rest_length))
        
        self.num_constraints = len(constraints)
        
        # Setup constraint arrays
        constraint_indices = np.zeros((self.num_constraints, 2), dtype=np.int32)
        constraint_rest_lengths = np.zeros(self.num_constraints, dtype=np.float32)
        
        for i, (idx1, idx2, rest_length) in enumerate(constraints):
            constraint_indices[i] = [idx1, idx2]
            constraint_rest_lengths[i] = rest_length
        
        self.constraint_indices = wp.from_numpy(constraint_indices, dtype=wp.vec2i)
        self.constraint_rest_lengths = wp.from_numpy(constraint_rest_lengths, dtype=float)
        self.constraint_count = wp.array([self.num_constraints], dtype=int)
        
        # Bending constraints (simplified)
        self.bend_count = wp.array([0], dtype=int)
    
    def _setup_wind_sources(self):
        """Setup wind source arrays for multiple wind sources"""
        # Create arrays for wind sources
        wind_centers = []
        wind_strengths = []
        wind_radii = []
        wind_directions = []
        wind_rotation_speeds = []
        
        for wind_source in self.wind_sources:
            wind_centers.append(wind_source['center'])
            wind_strengths.append(wind_source['strength'])
            wind_radii.append(wind_source['radius'])
            wind_directions.append(wind_source['direction'])
            wind_rotation_speeds.append(wind_source['rotation_speed'])
        
        # Convert to Warp arrays
        self.wind_centers = wp.from_numpy(np.array(wind_centers), dtype=wp.vec3)
        self.wind_strengths = wp.from_numpy(np.array(wind_strengths), dtype=float)
        self.wind_radii = wp.from_numpy(np.array(wind_radii), dtype=float)
        self.wind_directions = wp.from_numpy(np.array(wind_directions), dtype=float)
        self.wind_rotation_speeds = wp.from_numpy(np.array(wind_rotation_speeds), dtype=float)
    
    def step(self, frame):
        """Perform one simulation step"""
        # Release corners after specified frame for more natural motion
        if frame >= self.corner_release_frame and not self.corners_released:
            # Release corner particles by setting their inverse mass to normal value
            corner_indices = [0, self.ny-1, (self.nx-1)*self.ny, (self.nx-1)*self.ny + self.ny-1]
            inv_mass_host = self.inv_mass.numpy()
            for idx in corner_indices:
                inv_mass_host[idx] = 1.0
            self.inv_mass = wp.from_numpy(inv_mass_host, dtype=float)
            self.corners_released = True
            print(f"  Released corner particles at frame {frame}")
        
        # Clear forces
        wp.launch(kernel=clear_forces, dim=self.num_particles, inputs=[self.f], device=self.x.device)
        
        # Apply overall air resistance
        wp.launch(
            kernel=apply_air_resistance,
            dim=self.num_particles,
            inputs=[self.v, self.f, self.air_resistance_coefficient, self.linear_damping],
            device=self.x.device
        )
        
        # Apply aerodynamic forces with enhanced wind noise
        wp.launch(
            kernel=apply_aerodynamic_forces,
            dim=self.num_particles,
            inputs=[self.x, self.v, self.f, self.inv_mass, self.base_wind_velocity,
                   self.air_density, self.drag_coefficient, self.lift_coefficient,
                   self.wind_noise_scale, self.wind_noise_strength, self.wind_turbulence,
                   frame * self.dt, self.nx, self.ny],
            device=self.x.device
        )
        
        # Apply side wind effects
        wp.launch(
            kernel=apply_side_wind,
            dim=self.num_particles,
            inputs=[self.x, self.f, self.side_wind_strength, self.side_wind_direction, frame * self.dt],
            device=self.x.device
        )
        
        # Apply omnidirectional wind from all sides
        wp.launch(
            kernel=apply_omnidirectional_wind,
            dim=self.num_particles,
            inputs=[self.x, self.f, self.omnidirectional_wind_strength, frame * self.dt],
            device=self.x.device
        )
        
        # Apply multiple rotating wind sources
        wp.launch(
            kernel=apply_multiple_wind_sources,
            dim=self.num_particles,
            inputs=[self.x, self.f, self.wind_centers, self.wind_strengths, self.wind_radii,
                   self.wind_directions, self.wind_rotation_speeds, self.wind_noise_scale,
                   self.wind_noise_strength, self.wind_turbulence, frame * self.dt, self.num_wind_sources],
            device=self.x.device
        )
        
        # Apply enhanced wind field effects
        wp.launch(
            kernel=apply_wind_field,
            dim=self.num_particles,
            inputs=[self.x, self.f, self.wind_center, self.wind_strength, 
                   self.wind_radius, self.wind_noise_scale, self.wind_noise_strength,
                   self.wind_turbulence, frame * self.dt],
            device=self.x.device
        )
        
        # Integrate particles
        wp.launch(
            kernel=integrate_particles,
            dim=self.num_particles,
            inputs=[self.x, self.v, self.f, self.inv_mass, self.dt, self.gravity],
            device=self.x.device
        )
        
        # Apply constraints (more iterations to prevent stretching with higher stiffness)
        for _ in range(10):  # Increased iterations for non-stretchable behavior
            wp.launch(
                kernel=apply_distance_constraints,
                dim=self.num_constraints,
                inputs=[self.x, self.inv_mass, self.constraint_count, 
                       self.constraint_indices, self.constraint_rest_lengths, 
                       self.stiffness, 10],  # More constraint iterations per step
                device=self.x.device
            )
        
        # Apply ground collision
        wp.launch(
            kernel=apply_ground_collision,
            dim=self.num_particles,
            inputs=[self.x, self.v, self.ground_height, self.thickness, self.restitution, self.friction],
            device=self.x.device
        )
        
        # Apply damping
        self.v = self.v * self.damping
    
    def check_landing(self):
        """Check if bedsheet has landed and settled"""
        # Check landing status for all particles
        wp.launch(
            kernel=check_landing_status,
            dim=self.num_particles,
            inputs=[self.x, self.v, self.ground_height, self.landing_threshold,
                   self.velocity_threshold, self.landing_status],
            device=self.x.device
        )
        
        # Count landed particles
        landing_status_host = self.landing_status.numpy()
        landed_count = np.sum(landing_status_host)
        landing_percentage = landed_count / self.num_particles
        
        return landing_percentage >= self.landing_percentage_threshold
    
    def get_positions(self):
        """Get current particle positions as numpy array"""
        return self.x.numpy()
    
    def get_mesh_data(self):
        """Get mesh data for Blender import"""
        positions = self.get_positions()
        
        # Create faces for cloth mesh
        faces = []
        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                # Get vertex indices
                v1 = i * self.ny + j
                v2 = (i + 1) * self.ny + j
                v3 = (i + 1) * self.ny + (j + 1)
                v4 = i * self.ny + (j + 1)
                
                # Create two triangles
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        return positions, faces


def save_mesh_data(positions, faces, output_path, nx=None, ny=None):
    """Save mesh data in a format that can be imported to Blender"""
    mesh_data = {
        'vertices': positions.tolist(),
        'faces': faces,
        'grid_size': [nx, ny] if nx and ny else None,
        'metadata': {
            'num_vertices': len(positions),
            'num_faces': len(faces),
            'bounds': {
                'min': positions.min(axis=0).tolist(),
                'max': positions.max(axis=0).tolist()
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(mesh_data, f, indent=2)
    
    print(f"Mesh data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Warp-based bedsheet simulation with aerodynamic flow')
    parser.add_argument('--width', type=float, default=2.0, help='Bedsheet width (m)')
    parser.add_argument('--height', type=float, default=1.0, help='Bedsheet height (m)')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution (higher = more vertices)')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum simulation steps')
    parser.add_argument('--output', default='enhanced_warp_bedsheet_output', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = EnhancedWarpBedsheetSimulator(
        width=args.width,
        height=args.height,
        resolution=args.resolution
    )
    
    print(f"Running enhanced simulation (max {args.max_steps} steps)...")
    
    # Run simulation until bedsheet lands or max steps reached
    for step in range(args.max_steps):
        simulator.step(step)
        
        # Save mesh data every 10 steps
        if step % 10 == 0:
            positions, faces = simulator.get_mesh_data()
            output_path = os.path.join(args.output, f'bedsheet_{step:04d}.json')
            save_mesh_data(positions, faces, output_path, simulator.nx, simulator.ny)
            
            if step % 50 == 0:
                print(f"Step {step}/{args.max_steps} completed")
        
        # Check if bedsheet has landed
        if step > 50:  # Give some time for initial settling
            if simulator.check_landing():
                print(f"Bedsheet has landed and settled at step {step}")
                break
    
    # Save final mesh
    positions, faces = simulator.get_mesh_data()
    final_path = os.path.join(args.output, 'bedsheet_final.json')
    save_mesh_data(positions, faces, final_path, simulator.nx, simulator.ny)
    
    print(f"Simulation completed! Mesh data saved to {args.output}")


if __name__ == '__main__':
    main()
