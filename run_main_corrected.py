"""
Main test with CORRECTED parameters (alpha=0)
"""

import sys
sys.path.insert(0, '.')

from experiments.main_test import run_main_test
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

print("\n" + "="*70)
print("MAIN TEST with CORRECTED PARAMETERS")
print("α=0 (no geometric energy), β=0.1, λ=100")
print("="*70 + "\n")

# Key parameters
N_values = [100, 200, 500, 1000, 2000, 5000]

df = run_main_test(
    N_values, 
    alpha=0.0,     # ← CORRECTED
    beta=0.1,
    lam=100.0,
    random_seed=42
)

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(df[['N', 'epsilon_opt', 'd_participation', 'd_MDS', 'S_ratio']].to_string(index=False))

# Save
df.to_csv('results_corrected.csv', index=False)
print("\nSaved to: results_corrected.csv")