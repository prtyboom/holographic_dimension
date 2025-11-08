"""
Holographic System: Core class for 4D→3D projection under holographic constraint.
FIXED: Entropy now depends on epsilon (effective rank).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import eigvalsh


class HolographicSystem:
    """
    System of N bits in 4D latent space with holographic projection to 3D.
    """
    
    def __init__(self, N, d_latent=4, alpha=1.0, beta=0.5, lam=100.0, 
                 sigma='auto', random_seed=None):
        
        self.N = N
        self.d_latent = d_latent
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Latent 4D coordinates
        self.Z = np.random.randn(N, d_latent)
        
        # Fix sigma by epsilon=1 (data scale)
        if sigma == 'auto':
            D2 = self._pairwise_dist2(epsilon=1.0)
            D = np.sqrt(np.clip(D2, 0.0, None))
            self.sigma = np.median(D[D > 0]) + 1e-6
        else:
            self.sigma = float(sigma)
        
        # Holographic bound
        self.S_max = N**(2.0/3.0) / 4.0
    
    def _pairwise_dist2(self, epsilon):
        """Pairwise squared distances with anisotropic weights."""
        weights = np.ones(self.d_latent)
        weights[-1] = epsilon**2
        diffs = self.Z[:, None, :] - self.Z[None, :, :]
        return np.sum((diffs**2) * weights, axis=2)
    
    def _project_spd_corr(self, C):
        """Project to SPD correlation matrix with diag=1."""
        # Symmetrize
        C = (C + C.T) / 2.0
        
        # Force positive eigenvalues
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        C = (V * w) @ V.T
        
        # Normalize diagonal to 1
        d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
        Dinv = np.diag(1.0 / d)
        C = Dinv @ C @ Dinv
        
        # Clip correlations to valid range
        C = np.clip(C, -1.0 + 1e-6, 1.0 - 1e-6)
        np.fill_diagonal(C, 1.0)
        
        return C
    
    def correlation_matrix(self, epsilon):
        """
        Build correlation matrix with fixed sigma scale.
        
        C_ij = exp(-(D_ij / σ)²)
        
        where D_ij depends on epsilon via anisotropic weights.
        """
        D2 = self._pairwise_dist2(epsilon)
        D = np.sqrt(np.clip(D2, 0.0, None))
        
        # Gaussian kernel with FIXED sigma
        C = np.exp(-(D / self.sigma)**2)
        
        # Project to valid correlation matrix
        return self._project_spd_corr(C)
    
    def entropy(self, C):
        """
        Effective rank (participation ratio).
        
        S_eff = (Σλᵢ)² / Σλᵢ²
        
        This DECREASES when epsilon decreases (4th axis suppressed).
        """
        w = eigvalsh((C + C.T) / 2.0)
        w = np.clip(w, 1e-12, None)
        
        sum_w = np.sum(w)
        sum_w2 = np.sum(w**2)
        
        if sum_w2 < 1e-15:
            return 0.0
        
        r_eff = (sum_w**2) / sum_w2
        
        return float(r_eff)
    
    def _U_laplacian(self, C):
        """Geometric energy: graph Laplacian."""
        D = np.diag(np.sum(C, axis=1))
        L = D - C
        U = np.trace(L @ L) / self.N
        return U
    
    def _U_volume(self, epsilon):
        """Thermodynamic energy: anti-collapse."""
        sigma_4 = np.std(self.Z[:, -1])
        V_4th = sigma_4 * epsilon
        reg = 1e-3
        U = 1.0 / (V_4th + reg)
        return U
    
    def _compute_internal_energy(self, C, epsilon):
        """U = α·U_geom + β·U_therm"""
        U_geom = self._U_laplacian(C)
        U_therm = self._U_volume(epsilon)
        return self.alpha * U_geom + self.beta * U_therm
    
    def _compute_penalty(self, S):
        """Holographic penalty: λ·max(0, S - S_max)²"""
        if S > self.S_max:
            return self.lam * (S - self.S_max)**2
        else:
            return 0.0
    
    def free_energy(self, epsilon):
        """F(ε) = U(ε) + penalty(S(ε))"""
        C = self.correlation_matrix(epsilon)
        S = self.entropy(C)
        U = self._compute_internal_energy(C, epsilon)
        penalty = self._compute_penalty(S)
        
        F = U + penalty
        return F
    
    def optimize_epsilon(self, bounds=(1e-6, 1.0), tol=1e-8, maxiter=500):
        """Find ε* = argmin F(ε)"""
        result = minimize_scalar(
            self.free_energy,
            bounds=bounds,
            method='bounded',
            options={'xatol': tol, 'maxiter': maxiter}
        )
        
        if not result.success:
            import warnings
            warnings.warn(f"Optimization failed: {result.message}")
        
        return result.x, result.fun
    
    def debug_S_vs_eps(self):
        """Debug: show how S changes with epsilon."""
        print("\nDEBUG: S_eff vs epsilon")
        print("=" * 60)
        for e in [1.0, 0.3, 0.1, 0.03, 0.01]:
            C = self.correlation_matrix(e)
            S = self.entropy(C)
            min_eig = eigvalsh(C)[0]
            print(f"ε={e:6.3f}  S_eff={S:8.3f}  S_max={self.S_max:8.3f}  "
                  f"ratio={S/self.S_max:6.3f}  min_eig={min_eig:.2e}")
        print("=" * 60)
    
    def __repr__(self):
        return f"HolographicSystem(N={self.N}, S_max={self.S_max:.2f}, σ={self.sigma:.3f})"