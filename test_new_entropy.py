"""
Test NEW entropy (Tr(C²)) - should scale with N!
"""

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem

print("="*70)
print("CRITICAL TEST: Does S scale with N now?")
print("="*70)

# Test different N
for N in [100, 500, 1000, 2000]:
    print(f"\n{'='*70}")
    system = HolographicSystem(N=N, d_latent=4, sigma='auto', 
                               alpha=1.0, beta=0.1, lam=100.0,
                               random_seed=42)
    
    print(f"N = {N:5d}, S_max = {system.S_max:8.2f}, σ = {system.sigma:.3f}")
    print("-"*70)
    print(f"{'ε':>8}  {'S=Tr(C²)':>12}  {'S_max':>10}  {'S/S_max':>10}  {'Penalty':>12}")
    print("-"*70)
    
    for eps in [1.0, 0.5, 0.1, 0.01]:
        C = system.correlation_matrix(eps)
        S = system.entropy(C)
        
        penalty = system.lam * max(0, S - system.S_max)**2
        
        print(f"{eps:8.3f}  {S:12.2f}  {system.S_max:10.2f}  "
              f"{S/system.S_max:10.3f}  {penalty:12.2e}")

print("\n" + "="*70)
print("KEY QUESTION: Does S at ε=1.0 GROW with N?")
print("="*70)