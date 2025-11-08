"""
Main Test: ε*(N) → 0 as N → ∞

Tests the central prediction that optimal epsilon decreases
with system size following a power law: ε*(N) ~ N^(-β)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

import sys
sys.path.insert(0, '.')

from core.system import HolographicSystem
from core.dimensions import dimension_participation, dimension_MDS


logger = logging.getLogger(__name__)


def run_main_test(N_values, random_seed=42, **system_params):
    """
    Run main experiment: optimize epsilon for different system sizes.
    
    Parameters
    ----------
    N_values : list of int
        System sizes to test
    random_seed : int, default=42
        Random seed for reproducibility
    **system_params : dict
        Additional parameters for HolographicSystem
        (alpha, beta, lam, sigma, etc.)
    
    Returns
    -------
    results : pandas.DataFrame
        Columns: N, epsilon_opt, F_opt, d_participation, d_MDS,
                 S, S_max, S_ratio, U, penalty
    """
    
    logger.info("=" * 60)
    logger.info("MAIN TEST: ε*(N) → 0")
    logger.info("=" * 60)
    logger.info(f"System sizes: {N_values}")
    logger.info(f"Parameters: {system_params}")
    logger.info("")
    
    results = []
    
    for N in tqdm(N_values, desc="Main Test", unit="system"):
        
        # Create system
        system = HolographicSystem(
            N=N, 
            random_seed=random_seed,
            **system_params
        )
        
        # Optimize epsilon
        epsilon_opt, F_opt = system.optimize_epsilon()
        
        # Compute observables at optimum
        C_opt = system.correlation_matrix(epsilon_opt)
        S_opt = system.entropy(C_opt)
        U_opt = system._compute_internal_energy(C_opt, epsilon_opt)
        penalty_opt = system._compute_penalty(S_opt)
        
        # Measure dimensions
        d_PR = dimension_participation(C_opt)
        d_MDS = dimension_MDS(C_opt, variance_threshold=0.95)
        
        # Store results
        results.append({
            'N': N,
            'epsilon_opt': epsilon_opt,
            'F_opt': F_opt,
            'd_participation': d_PR,
            'd_MDS': d_MDS,
            'S': S_opt,
            'S_max': system.S_max,
            'S_ratio': S_opt / system.S_max,
            'U': U_opt,
            'penalty': penalty_opt
        })
        
        logger.info(
            f"N={N:5d} | ε*={epsilon_opt:.6f} | "
            f"d_PR={d_PR:.2f} | d_MDS={d_MDS} | "
            f"S/S_max={S_opt/system.S_max:.3f}"
        )
    
    df = pd.DataFrame(results)
    
    logger.info("")
    logger.info("Main test completed")
    logger.info("=" * 60)
    
    return df


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    N_values = [100, 500, 1000]
    df = run_main_test(N_values)
    
    print("\nResults:")
    print(df.to_string(index=False))