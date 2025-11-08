"""
Test optimization: will ε* < 1.0 now?
"""

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem
from core.dimensions import dimension_participation, dimension_MDS

print("="*70)
print("OPTIMIZATION TEST: Finding ε* for N=1000")
print("="*70)

system = HolographicSystem(N=1000, d_latent=4, sigma='auto',
                           alpha=1.0, beta=0.1, lam=100.0,
                           random_seed=42)

print(f"\nSystem: {system}")
print(f"S_max = {system.S_max:.2f}")
print()

# Debug scan first
system.debug_S_vs_eps()

# Optimize
print("\n" + "="*70)
print("Running optimization...")
print("="*70)

epsilon_opt, F_opt = system.optimize_epsilon()

print(f"\n{'='*70}")
print("RESULT:")
print("="*70)
print(f"Optimal ε* = {epsilon_opt:.6f}")
print(f"Free energy F(ε*) = {F_opt:.2e}")

# Observables at optimum
C_opt = system.correlation_matrix(epsilon_opt)
S_opt = system.entropy(C_opt)
d_PR = dimension_participation(C_opt)
d_MDS = dimension_MDS(C_opt)

print(f"\nAt ε* = {epsilon_opt:.6f}:")
print(f"  S = {S_opt:.2f}")
print(f"  S/S_max = {S_opt/system.S_max:.3f}")
print(f"  d_participation = {d_PR:.2f}")
print(f"  d_MDS = {d_MDS}")
print("="*70)

if epsilon_opt < 0.9:
    print("\n🎉 SUCCESS! ε* < 1.0 — Fourth dimension is being suppressed!")
else:
    print("\n⚠️  ε* still near 1.0 — Need to adjust parameters")