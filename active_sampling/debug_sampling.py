"""
Debug script focused on active sampling algorithm testing.

Tests the sample_gaussian_posterior() function on:
1. Structured Gaussian channel
2. Gaussian spikes channel

Allows tuning xi parameter for guidance strength.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from gaussian_test import (
    generate_structured_gaussian_channel, 
    generate_gaussian_spikes_channel,
    create_diffusion_schedule,
    initialize_params,
    sample_gaussian_posterior
)

# Test parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_a, N_c = 32, 64

print("=" * 80)
print("Active Sampling Algorithm Debug")
print("=" * 80)

# ============================================================================
# Test 1: Active Sampling on Structured Gaussian Channel
# ============================================================================
print("\n[Test 1] Active Sampling on Structured Gaussian Channel")
print("-" * 80)

h_true_structured, Sigma_structured = generate_structured_gaussian_channel(
    N_a, N_c, rho_ant=0.7, rho_car=0.7, device=device
)
Sigma_structured_inv = torch.linalg.inv(Sigma_structured)

print(f"Channel norm: {torch.norm(h_true_structured, p='fro'):.4f}")
print(f"Covariance condition number: {torch.linalg.cond(Sigma_structured).item():.2e}")

# Initialize parameters
alphas, alphas_cumprod = create_diffusion_schedule(device=device)
sigma_H, delta, t_0, Y_0, x_tau, pilot_indices, data_indices = \
    initialize_params(h_true_structured, N_p=4, gamma=14.4, sigma_n_sq=0.01, 
                      alphas_cumprod=alphas, device=device)

print(f"Initial pilots: {pilot_indices.tolist()}")
print(f"Data subcarriers: {len(data_indices)}")
print(f"Starting diffusion at t_0={t_0}, sigma_H={sigma_H:.4f}, delta={delta:.4f}")

# Run active sampling
print("\nRunning active sampling with xi=0.1...")
try:
    particles_struct, history_struct = sample_gaussian_posterior(
        Y_0, Sigma_structured, Sigma_structured_inv,
        pilot_indices, data_indices,
        N_p=4,
        N_gen=150,
        xi=0.1,
        gamma=14.4,
        sigma_n_sq=0.01,
        selection_freq=50,
        device=device
    )
    
    print(f"\n✓ Active sampling completed!")
    print(f"  Selected pilots: {history_struct['pilots_selected']}")
    print(f"  Total pilots: {len(pilot_indices) + len(history_struct['pilots_selected'])}")
    
    # Reconstruction error
    errors_struct = []
    for i in range(4):
        error = torch.norm(particles_struct[i] - h_true_structured, p='fro') / \
                torch.norm(h_true_structured, p='fro')
        errors_struct.append(error.item())
        print(f"  Particle {i} error: {error:.6f}")
    
    print(f"  Mean error: {np.mean(errors_struct):.6f}")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {str(e)[:200]}")
    errors_struct = None

# ============================================================================
# Test 2: Active Sampling on Gaussian Spikes Channel
# ============================================================================
print("\n[Test 2] Active Sampling on Gaussian Spikes Channel")
print("-" * 80)

h_true_spikes, Sigma_spikes = generate_gaussian_spikes_channel(
    N_a, N_c, spike_indices=[16, 32, 48], device=device
)
Sigma_spikes_inv = torch.linalg.inv(Sigma_spikes)

print(f"Channel norm: {torch.norm(h_true_spikes, p='fro'):.4f}")
print(f"Covariance condition number: {torch.linalg.cond(Sigma_spikes).item():.2e}")
print(f"Spike locations: [16, 32, 48]")

# Initialize parameters
sigma_H, delta, t_0, Y_0, x_tau, pilot_indices, data_indices = \
    initialize_params(h_true_spikes, N_p=4, gamma=14.4, sigma_n_sq=0.01, 
                      alphas_cumprod=alphas, device=device)

print(f"Initial pilots: {pilot_indices.tolist()}")
print(f"Data subcarriers: {len(data_indices)}")
print(f"Starting diffusion at t_0={t_0}, sigma_H={sigma_H:.4f}, delta={delta:.4f}")

# Try different xi values
xi_values = [0.01, 0.05, 0.1]
results_spikes = {}

for xi_test in xi_values:
    print(f"\nRunning active sampling with xi={xi_test}...")
    try:
        particles_spikes, history_spikes = sample_gaussian_posterior(
            Y_0, Sigma_spikes, Sigma_spikes_inv,
            pilot_indices, data_indices,
            N_p=4,
            N_gen=150,
            xi=xi_test,
            gamma=14.4,
            sigma_n_sq=0.01,
            selection_freq=50,
            device=device
        )
        
        # Analyze results
        selected_pilots = history_spikes['pilots_selected']
        print(f"  ✓ Selected pilots: {selected_pilots}")
        print(f"    Total pilots: {len(pilot_indices) + len(selected_pilots)}")
        
        # Distance to nearest spike
        spike_locs = [16, 32, 48]
        distances = []
        for p in selected_pilots:
            dist_to_spike = min([abs(p - s) for s in spike_locs])
            distances.append(dist_to_spike)
        
        if distances:
            avg_dist = np.mean(distances)
            print(f"    Avg distance to nearest spike: {avg_dist:.2f} subcarriers")
            if avg_dist < 5:
                print(f"    ✓ Algorithm found the spikes!")
            else:
                print(f"    ⚠ Algorithm may need better guidance (try different xi)")
        
        # Reconstruction errors
        errors_spikes = []
        for i in range(4):
            error = torch.norm(particles_spikes[i] - h_true_spikes, p='fro') / \
                    torch.norm(h_true_spikes, p='fro')
            errors_spikes.append(error.item())
        
        print(f"    Mean particle error: {np.mean(errors_spikes):.6f}")
        
        results_spikes[xi_test] = {
            'particles': particles_spikes,
            'history': history_spikes,
            'errors': errors_spikes,
            'distances': distances
        }
        
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {str(e)[:200]}")

# ============================================================================
# Summary and recommendations
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if errors_struct is not None:
    print(f"\n[Structured Channel]")
    print(f"  Mean reconstruction error: {np.mean(errors_struct):.6f}")

print(f"\n[Gaussian Spikes Channel]")
if results_spikes:
    for xi_val, result in results_spikes.items():
        avg_error = np.mean(result['errors'])
        avg_dist = np.mean(result['distances']) if result['distances'] else np.inf
        print(f"\n  xi={xi_val}:")
        print(f"    Reconstruction error: {avg_error:.6f}")
        print(f"    Distance to spikes: {avg_dist:.2f} subcarriers")
        if avg_dist < 5:
            print(f"    ✓ GOOD: Algorithm finds spikes")
        else:
            print(f"    ⚠ NEEDS TUNING: Try adjusting xi")
else:
    print("  (No results - check for errors above)")

print("\n" + "=" * 80)
print("Tips for debugging:")
print("  - Increase xi if particles diverge or error grows")
print("  - Decrease xi if guidance is too aggressive")
print("  - Check final disagreement values to see if algorithm is uncertain")
print("=" * 80)
