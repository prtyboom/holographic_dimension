"""
Dimensionality Measures

Functions to compute effective dimensionality of correlation structures.
Implements:
- Participation Ratio (continuous measure)
- MDS-based dimension (threshold-based)
"""

import numpy as np
from scipy.linalg import eigvalsh


def dimension_participation(C):
    """
    Participation ratio: effective number of active modes.
    
    Formula:
        d_eff = (Σλᵢ)² / Σλᵢ²
    
    This is a continuous measure without arbitrary thresholds.
    
    Parameters
    ----------
    C : ndarray, shape (N, N)
        Correlation or covariance matrix
    
    Returns
    -------
    d_eff : float
        Effective dimensionality (participation ratio)
    
    References
    ----------
    - Used in quantum entanglement theory
    - Related to inverse participation ratio in localization
    """
    # Compute eigenvalues
    eigenvalues = eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # filter noise
    
    # Participation ratio
    sum_eigs = np.sum(eigenvalues)
    sum_eigs_sq = np.sum(eigenvalues**2)
    
    if sum_eigs_sq < 1e-15:
        return 0.0
    
    d_eff = sum_eigs**2 / sum_eigs_sq
    
    return d_eff


def dimension_MDS(C, variance_threshold=0.95):
    """
    Multidimensional Scaling dimension.
    
    Number of principal components needed to explain
    `variance_threshold` fraction of total variance.
    
    Parameters
    ----------
    C : ndarray, shape (N, N)
        Correlation matrix
    variance_threshold : float, default=0.95
        Fraction of variance to explain (0 < threshold < 1)
    
    Returns
    -------
    d_eff : int
        Number of effective dimensions
    
    Notes
    -----
    This gives discrete (integer) results, unlike participation ratio.
    Standard choice: 95% variance threshold.
    """
    # Compute eigenvalues (sorted descending)
    eigenvalues = eigvalsh(C)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Cumulative explained variance
    cumsum = np.cumsum(eigenvalues)
    total = np.sum(eigenvalues)
    
    if total < 1e-15:
        return 0
    
    explained_variance = cumsum / total
    
    # Find number of components for threshold
    d_eff = np.searchsorted(explained_variance, variance_threshold) + 1
    
    # Cap at maximum possible
    d_eff = min(d_eff, len(eigenvalues))
    
    return int(d_eff)


def get_eigenvalue_spectrum(C):
    """
    Get full eigenvalue spectrum (for diagnostic plots).
    
    Parameters
    ----------
    C : ndarray, shape (N, N)
        Correlation matrix
    
    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues in descending order
    """
    eigenvalues = eigvalsh(C)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    return eigenvalues