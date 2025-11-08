"""
Test with large N to see holographic pressure activate
"""

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem

print("="*60)
print("LARGE N TEST: N = 5000")
print("="*60)

system = HolographicSystem(N=5000, d_latent=4, sigma='auto', 
                           alpha=1.0, beta=0.1, lam=100.0, 
                           random_seed=42)

print(f"\nSystem: {system}")
print(f"S_max = {system.S_max:.2f}")
print()

# Manual scan of epsilon
print("Manual scan of S(ε):")
print("-"*60)
print(f"{'ε':>8}  {'S_eff':>8}  {'S_max':>8}  {'S/S_max':>8}  {'Penalty':>12}")
print("-"*60)

eps_vals = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

for eps in eps_vals:
    C = system.correlation_matrix(eps)
    S = system.entropy(C)
    penalty = system._compute_penalty(S)
    
    print(f"{eps:8.3f}  {S:8.2f}  {system.S_max:8.2f}  "
          f"{S/system.S_max:8.3f}  {penalty:12.2e}")

print("-"*60)
print()

# Now optimize
print("Running optimization...")
epsilon_opt, F_opt = system.optimize_epsilon()

C_opt = system.correlation_matrix(epsilon_opt)
S_opt = system.entropy(C_opt)

print(f"\nOptimal ε* = {epsilon_opt:.6f}")
print(f"S_eff(ε*) = {S_opt:.2f}")
print(f"S/S_max = {S_opt/system.S_max:.3f}")
print(f"F(ε*) = {F_opt:.2e}")
print()

# Dimension
from core.dimensions import dimension_participation, dimension_MDS

d_PR = dimension_participation(C_opt)
d_MDS = dimension_MDS(C_opt)

print(f"d_participation = {d_PR:.2f}")
print(f"d_MDS = {d_MDS}")
print("="*60)