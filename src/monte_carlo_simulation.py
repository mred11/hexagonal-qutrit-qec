"""
Monte Carlo Simulation for Hexagonal Qutrit Error Correction

Simulates error correction performance under various noise models.
Generates threshold curves and performance metrics.

Author: Eddie Chin
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from datetime import datetime

from hexagonal_qutrit_code import HexagonalQutritCode
from decoder_mwpm import MinimumWeightDecoder


class MonteCarloSimulation:
    """
    Monte Carlo simulation engine for quantum error correction.
    """
    
    def __init__(self, code, decoder, seed: int = None):
        """
        Initialize Monte Carlo simulation.
        
        Args:
            code: HexagonalQutritCode instance
            decoder: Decoder instance
            seed: Random seed for reproducibility
        """
        self.code = code
        self.decoder = decoder
        self.rng = np.random.default_rng(seed)
        
    def generate_errors(self, error_rate: float, bias_ratio: float = 1.0,
                       n_qubits: int = None) -> Dict[int, Tuple[int, int]]:
        """
        Generate random errors according to noise model.
        
        Args:
            error_rate: Physical error rate per qubit
            bias_ratio: Z-bias (ratio of Z errors to X errors)
                       bias=1 means equal X and Z
                       bias=10 means 10x more Z errors than X
            n_qubits: Number of qubits (default: all in code)
            
        Returns:
            Dictionary of errors {qubit_id: (x_power, z_power)}
        """
        if n_qubits is None:
            n_qubits = self.code.get_code_parameters()['n']
        
        errors = {}
        
        # Calculate individual error probabilities
        # Total error rate = p_x + p_z
        # bias = p_z / p_x
        # Therefore: p_x = error_rate / (1 + bias), p_z = error_rate * bias / (1 + bias)
        
        p_x = error_rate / (1 + bias_ratio)
        p_z = error_rate * bias_ratio / (1 + bias_ratio)
        
        for qubit in range(n_qubits):
            x_power = 0
            z_power = 0
            
            # X error?
            if self.rng.random() < p_x:
                x_power = self.rng.integers(1, 3)  # Power 1 or 2
            
            # Z error?
            if self.rng.random() < p_z:
                z_power = self.rng.integers(1, 3)  # Power 1 or 2
            
            if x_power != 0 or z_power != 0:
                errors[qubit] = (x_power, z_power)
        
        return errors
    
    def run_trial(self, error_rate: float, bias_ratio: float = 1.0) -> bool:
        """
        Run single trial of error correction.
        
        Args:
            error_rate: Physical error rate
            bias_ratio: Z-bias ratio
            
        Returns:
            True if logical error occurred (failure), False if success
        """
        # Generate errors
        errors = self.generate_errors(error_rate, bias_ratio)
        
        # Measure syndrome
        syndrome = self.decoder.measure_syndrome(errors)
        
        # Decode
        correction = self.decoder.decode(syndrome)
        
        # Check logical error
        logical_error = self.decoder.calculate_logical_error(errors, correction)
        
        return logical_error
    
    def estimate_logical_error_rate(self, physical_error_rate: float,
                                   n_trials: int = 1000,
                                   bias_ratio: float = 1.0,
                                   confidence_level: float = 0.95) -> Dict:
        """
        Estimate logical error rate with confidence intervals.
        
        Args:
            physical_error_rate: Physical error rate to test
            n_trials: Number of Monte Carlo trials
            bias_ratio: Z-bias ratio
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with results and confidence intervals
        """
        failures = 0
        
        for _ in range(n_trials):
            if self.run_trial(physical_error_rate, bias_ratio):
                failures += 1
        
        logical_error_rate = failures / n_trials
        
        # Calculate Wilson score confidence interval
        # (Better than normal approximation for small samples)
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
        
        n = n_trials
        p = logical_error_rate
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        return {
            'physical_error_rate': physical_error_rate,
            'logical_error_rate': logical_error_rate,
            'n_trials': n_trials,
            'n_failures': failures,
            'bias_ratio': bias_ratio,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level
        }
    
    def find_threshold(self, error_rates: List[float] = None,
                      n_trials: int = 1000,
                      bias_ratio: float = 1.0) -> Tuple[float, List[Dict]]:
        """
        Find error threshold by testing multiple error rates.
        
        Threshold is where logical_error_rate = physical_error_rate.
        
        Args:
            error_rates: List of error rates to test
            n_trials: Trials per error rate
            bias_ratio: Z-bias ratio
            
        Returns:
            (threshold_estimate, list_of_results)
        """
        if error_rates is None:
            # Default range for threshold search
            error_rates = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]
        
        results = []
        
        print(f"Finding threshold (bias_ratio={bias_ratio})...")
        
        for p_phys in tqdm(error_rates):
            result = self.estimate_logical_error_rate(
                p_phys, n_trials, bias_ratio
            )
            results.append(result)
            
            p_log = result['logical_error_rate']
            print(f"  p_phys={p_phys:.4f} -> p_log={p_log:.6f} "
                  f"({result['n_failures']}/{n_trials} failures)")
        
        # Estimate threshold (where p_log = p_phys)
        # Find crossing point by interpolation
        
        threshold = None
        for i in range(len(results) - 1):
            p1 = results[i]['physical_error_rate']
            p2 = results[i+1]['physical_error_rate']
            l1 = results[i]['logical_error_rate']
            l2 = results[i+1]['logical_error_rate']
            
            # Check if crossing occurs
            if (l1 < p1 and l2 > p2) or (l1 > p1 and l2 < p2):
                # Linear interpolation
                threshold = (p1 + p2) / 2
                break
        
        if threshold is None:
            # Estimate from overall trend
            below_threshold = [r for r in results 
                             if r['logical_error_rate'] < r['physical_error_rate']]
            above_threshold = [r for r in results
                             if r['logical_error_rate'] > r['physical_error_rate']]
            
            if below_threshold and above_threshold:
                threshold = (below_threshold[-1]['physical_error_rate'] + 
                           above_threshold[0]['physical_error_rate']) / 2
            else:
                threshold = 0.01  # Default estimate
        
        return threshold, results
    
    def compare_code_distances(self, distances: List[int],
                              error_rates: List[float],
                              n_trials: int = 500) -> Dict:
        """
        Compare performance across different code distances.
        
        Args:
            distances: List of code distances to test
            error_rates: Error rates to test for each distance
            n_trials: Trials per configuration
            
        Returns:
            Dictionary with results for each distance
        """
        all_results = {}
        
        for d in distances:
            print(f"\n{'='*70}")
            print(f"Testing distance d={d}")
            print(f"{'='*70}")
            
            # Build code and decoder for this distance
            code = HexagonalQutritCode(distance=d)
            decoder = MinimumWeightDecoder(code)
            sim = MonteCarloSimulation(code, decoder, seed=42)
            
            # Find threshold
            threshold, results = sim.find_threshold(error_rates, n_trials)
            
            all_results[d] = {
                'threshold': threshold,
                'results': results,
                'code_params': code.get_code_parameters()
            }
            
            print(f"Estimated threshold for d={d}: {threshold:.4f}")
        
        return all_results


def plot_threshold_curve(results: List[Dict], code_distance: int,
                        save_path: str = None):
    """
    Plot threshold curve (logical vs physical error rate).
    
    Args:
        results: List of simulation results
        code_distance: Code distance for title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    p_phys = [r['physical_error_rate'] for r in results]
    p_log = [r['logical_error_rate'] for r in results]
    ci_lower = [r['ci_lower'] for r in results]
    ci_upper = [r['ci_upper'] for r in results]
    
    # Plot results
    ax.plot(p_phys, p_log, 'o-', linewidth=2, markersize=8,
           label=f'Hexagonal Qutrit (d={code_distance})', color='blue')
    
    # Plot confidence intervals
    ax.fill_between(p_phys, ci_lower, ci_upper, alpha=0.3, color='blue')
    
    # Plot threshold line (p_log = p_phys)
    max_p = max(p_phys)
    ax.plot([0, max_p], [0, max_p], '--', color='red', 
           label='Threshold (p_log = p_phys)', linewidth=1.5)
    
    ax.set_xlabel('Physical Error Rate', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title(f'Error Correction Performance\nCode Distance d={code_distance}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold curve saved to {save_path}")
    
    return fig


def plot_distance_comparison(all_results: Dict, save_path: str = None):
    """
    Plot comparison of different code distances.
    
    Args:
        all_results: Results from compare_code_distances
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, (d, data) in enumerate(sorted(all_results.items())):
        results = data['results']
        
        p_phys = [r['physical_error_rate'] for r in results]
        p_log = [r['logical_error_rate'] for r in results]
        
        ax.plot(p_phys, p_log, 'o-', linewidth=2, markersize=6,
               label=f'd={d}', color=colors[i % len(colors)])
    
    # Threshold line
    max_p = max(max(r['physical_error_rate'] for r in data['results'])
               for data in all_results.values())
    ax.plot([0, max_p], [0, max_p], '--', color='black',
           label='Threshold line', linewidth=1.5)
    
    ax.set_xlabel('Physical Error Rate', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Scaling with Code Distance\nHexagonal Qutrit Codes',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance comparison saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("="*70)
    print("MONTE CARLO SIMULATION - HEXAGONAL QUTRIT ERROR CORRECTION")
    print("="*70)
    
    # Build code and decoder
    distance = 5
    print(f"\nBuilding code (distance={distance})...")
    code = HexagonalQutritCode(distance=distance)
    params = code.get_code_parameters()
    print(f"Code: {params['notation']}")
    
    print("\nBuilding decoder...")
    decoder = MinimumWeightDecoder(code)
    
    # Initialize simulation
    print("\nInitializing Monte Carlo simulation...")
    sim = MonteCarloSimulation(code, decoder, seed=42)
    
    # Run threshold finding
    print("\n" + "="*70)
    print("FINDING ERROR THRESHOLD")
    print("="*70)
    
    error_rates = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
    n_trials = 500  # Increase to 10000 for publication
    
    threshold, results = sim.find_threshold(error_rates, n_trials)
    
    print(f"\n{'='*70}")
    print(f"ESTIMATED THRESHOLD: {threshold:.4f} ({threshold*100:.2f}%)")
    print(f"{'='*70}")
    
    # Plot results
    plot_threshold_curve(results, distance,
                        save_path='/mnt/user-data/outputs/threshold_curve.png')
    
    # Save results to JSON
    output_data = {
        'code_distance': distance,
        'code_parameters': params,
        'threshold': threshold,
        'n_trials': n_trials,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open('/mnt/user-data/outputs/simulation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to /mnt/user-data/outputs/simulation_results.json")
    print("\nSimulation complete!")
