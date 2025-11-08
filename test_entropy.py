"""
Quick test: does S depend on epsilon?
"""

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem

# Create small system
print("Creating system N=500...")
system = HolographicSystem(N=500, d_latent=4, sigma='auto', random_seed=42)

print(f"System: {system}")
print(f"Sigma (fixed): {system.sigma:.4f}")
print()

# Debug: S vs epsilon
system.debug_S_vs_eps()