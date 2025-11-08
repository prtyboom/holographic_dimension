"""
Critical test: Remove geometric energy (alpha=0)
"""

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem
from core.dimensions import dimension_participation, dimension_MDS

print("="*70)
print("CRITICAL TEST: α=0 (no geometric energy)")
print("="*70)

system = HolographicSystem(
    N=1000, 
    d_latent=4,
    alpha=0.0,      # ← KEY: No U_geom
    beta=0.1,
    lam=100.0,
    sigma='auto',
    random_seed=42
)

print(f"System: {system}")
print(f"Parameters: α={system.alpha}, β={system.beta}, λ={system.lam}")
print()

# Optimize
print("Optimizing...")
epsilon_opt, F_opt = system.optimize_epsilon()

print("\n" + "="*70)
print("RESULT:")
print("="*70)
print(f"ε* = {epsilon_opt:.6f}")
print(f"F(ε*) = {F_opt:.2e}")

# Observables
C_opt = system.correlation_matrix(epsilon_opt)
S_opt = system.entropy(C_opt)
d_PR = dimension_participation(C_opt)
d_MDS = dimension_MDS(C_opt)

print(f"\nAt optimum:")
print(f"  S = {S_opt:.2f}")
print(f"  S/S_max = {S_opt/system.S_max:.3f}")
print(f"  d_participation = {d_PR:.2f}")
print(f"  d_MDS = {d_MDS}")
print("="*70)

if epsilon_opt < 0.5:
    print("\n🎉🎉🎉 SUCCESS! ε* << 1 — Holographic pressure works!")
    print("Fourth dimension is being suppressed!")
else:
    print(f"\n⚠️ ε* = {epsilon_opt:.3f} still high")