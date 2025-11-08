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
        Parameters for HolographicSystem (alpha, beta, lam, etc.)
    
    Returns
    -------
    results : pandas.DataFrame
        Columns: N, epsilon_opt, F_opt, d_participation, d_MDS,
                 S, S_max, S_ratio, U_geom, U_therm, U_total, penalty
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
        
        # Energy components
        U_geom = system._U_laplacian(C_opt)
        U_therm = system._U_volume(epsilon_opt)
        U_total = system.alpha * U_geom + system.beta * U_therm
        
        # Penalty
        if S_opt > system.S_max:
            penalty = system.lam * (S_opt - system.S_max)**2
        else:
            penalty = 0.0
        
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
            'U_geom': U_geom,
            'U_therm': U_therm,
            'U_total': U_total,
            'penalty': penalty
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
    
    # CORRECTED: alpha=0 (no geometric energy)
    N_values = [100, 500, 1000]
    df = run_main_test(
        N_values,
        alpha=0.0,
        beta=0.1,
        lam=100.0
    )
    
    print("\nResults:")
    print(df[['N', 'epsilon_opt', 'd_participation', 'S_ratio']].to_string(index=False))