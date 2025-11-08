"""
Visualization of main results: ε*(N) power law
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data from experiments (N=100-2000, confirmed)
N_values = np.array([100, 200, 500, 1000, 2000])
epsilon_values = np.array([0.003518, 0.002351, 0.001307, 0.000809, 0.000465])
d_PR_values = np.array([3.05, 3.13, 3.14, 3.19, 3.20])
S_ratio_values = np.array([14.213, 14.926, 15.458, 16.083, 16.511])

# Power law fit
def power_law(N, a, b):
    return a * N**b

params, cov = curve_fit(power_law, N_values, epsilon_values)
a_fit, b_fit = params
a_err, b_err = np.sqrt(np.diag(cov))

print(f"Power law fit: ε*(N) = {a_fit:.4f} · N^({b_fit:.3f})")
print(f"Errors: a = ±{a_err:.4f}, b = ±{b_err:.3f}")

# R² calculation
residuals = epsilon_values - power_law(N_values, *params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((epsilon_values - np.mean(epsilon_values))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R² = {r_squared:.6f}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: ε*(N) with power law fit
ax1 = axes[0, 0]
N_fit = np.logspace(2, 4, 100)  # 100 to 10000
epsilon_fit = power_law(N_fit, *params)

ax1.loglog(N_values, epsilon_values, 'ro', markersize=10, label='Data')
ax1.loglog(N_fit, epsilon_fit, 'b-', linewidth=2, 
           label=f'Fit: $\\varepsilon^* = {a_fit:.3f} \\cdot N^{{{b_fit:.2f}}}$\n$R^2 = {r_squared:.4f}$')
ax1.set_xlabel('System Size $N$', fontsize=12)
ax1.set_ylabel('Optimal $\\varepsilon^*$', fontsize=12)
ax1.set_title('(a) Projection Parameter Suppression', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: d_PR(N)
ax2 = axes[0, 1]
ax2.plot(N_values, d_PR_values, 'go-', markersize=8, linewidth=2)
ax2.axhline(y=3.0, color='r', linestyle='--', linewidth=1.5, label='Target $d=3$')
ax2.set_xlabel('System Size $N$', fontsize=12)
ax2.set_ylabel('Participation Ratio Dimension $d_{PR}$', fontsize=12)
ax2.set_title('(b) Effective Dimensionality', fontsize=13, fontweight='bold')
ax2.set_ylim([2.9, 3.3])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: S/S_max(N)
ax3 = axes[1, 0]
ax3.plot(N_values, S_ratio_values, 'mo-', markersize=8, linewidth=2)
ax3.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.5, label='Holographic Bound')
ax3.set_xlabel('System Size $N$', fontsize=12)
ax3.set_ylabel('$S / S_{max}$', fontsize=12)
ax3.set_title('(c) Holographic Pressure', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Suppression percentage
ax4 = axes[1, 1]
suppression_pct = (1 - epsilon_values) * 100
ax4.semilogx(N_values, suppression_pct, 'co-', markersize=8, linewidth=2)
ax4.set_xlabel('System Size $N$', fontsize=12)
ax4.set_ylabel('4th Dimension Suppression (\%)', fontsize=12)
ax4.set_title('(d) Observational Invisibility', fontsize=13, fontweight='bold')
ax4.set_ylim([99, 100])
ax4.grid(True, alpha=0.3)

# Add text annotation
ax4.text(500, 99.85, f'At $N=2000$:\n{suppression_pct[-1]:.2f}\% suppressed', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10)

plt.tight_layout()
plt.savefig('results/figures/main_results.png', dpi=300, bbox_inches='tight')
print("\nFigure saved to: results/figures/main_results.png")

plt.show()