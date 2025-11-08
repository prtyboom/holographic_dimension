"""
Debug: Why optimizer chooses ε=1 despite penalty?
"""

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem

system = HolographicSystem(N=1000, alpha=1.0, beta=0.1, lam=100.0, random_seed=42)

print("="*70)
print("ENERGY BALANCE: Why ε*=1.0?")
print("="*70)
print(f"N={system.N}, S_max={system.S_max:.2f}, σ={system.sigma:.3f}")
print(f"Parameters: α={system.alpha}, β={system.beta}, λ={system.lam}")
print("="*70)
print(f"{'ε':>8}  {'U_geom':>10}  {'U_therm':>10}  {'U_total':>10}  {'Penalty':>10}  {'F':>12}")
print("-"*70)

for eps in [1.0, 0.5, 0.3, 0.2, 0.1]:
    C = system.correlation_matrix(eps)
    S = system.entropy(C)
    
    U_geom = system._U_laplacian(C)
    U_therm = system._U_volume(eps)
    U = system.alpha * U_geom + system.beta * U_therm
    
    penalty = system.lam * max(0, S - system.S_max)**2
    
    F = U + penalty
    
    print(f"{eps:8.3f}  {U_geom:10.2f}  {U_therm:10.2f}  {U:10.2f}  {penalty:10.2f}  {F:12.2f}")

print("="*70)