"""
Basic Usage Examples

Simple examples demonstrating how to use the hexagonal qutrit
error correction code implementation.

Author: Eddie Chin
Date: November 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hexagonal_qutrit_code import HexagonalQutritCode
from decoder_mwpm import MinimumWeightDecoder
from monte_carlo_simulation import MonteCarloSimulation


def example_1_code_construction():
    """
    Example 1: Build a hexagonal qutrit code.
    """
    print("="*70)
    print("EXAMPLE 1: Code Construction")
    print("="*70)
    
    # Create code with distance 5
    code = HexagonalQutritCode(distance=5)
    
    # Get code parameters
    params = code.get_code_parameters()
    
    print(f"\nCode parameters: {params['notation']}")
    print(f"Physical qutrits: {params['n']}")
    print(f"Logical qutrits: {params['k']}")
    print(f"Code distance: {params['d']}")
    
    # Get stabilizer information
    print(f"\nStabilizers:")
    print(f"  X-type (detect Z errors): {len(code.stabilizers['X_type'])}")
    print(f"  Z-type (detect X errors): {len(code.stabilizers['Z_type'])}")
    
    print("\n✓ Code construction complete!")


def example_2_decoder():
    """
    Example 2: Build decoder and test error correction.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Decoder")
    print("="*70)
    
    # Build code and decoder
    code = HexagonalQutritCode(distance=3)
    decoder = MinimumWeightDecoder(code)
    
    print(f"\nBuilt decoder for {code.get_code_parameters()['notation']} code")
    
    # Simulate random errors
    import numpy as np
    np.random.seed(42)
    
    errors = {}
    n_qubits = code.get_code_parameters()['n']
    
    # Generate a few random errors
    for _ in range(3):
        qubit = np.random.randint(0, n_qubits)
        x_power = np.random.randint(1, 3)
        z_power = np.random.randint(0, 3)
        errors[qubit] = (x_power, z_power)
    
    print(f"\nGenerated {len(errors)} random errors")
    
    # Measure syndrome
    syndrome = decoder.measure_syndrome(errors)
    print(f"Syndrome: X-type={len(syndrome['X_type'])}, Z-type={len(syndrome['Z_type'])}")
    
    # Decode
    correction = decoder.decode(syndrome)
    print(f"Correction: {len(correction)} qubits")
    
    # Check if successful
    logical_error = decoder.calculate_logical_error(errors, correction)
    
    if logical_error:
        print("\n✗ Logical error occurred")
    else:
        print("\n✓ Error successfully corrected!")


def example_3_monte_carlo():
    """
    Example 3: Run Monte Carlo simulation.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Monte Carlo Simulation")
    print("="*70)
    
    # Setup
    code = HexagonalQutritCode(distance=5)
    decoder = MinimumWeightDecoder(code)
    sim = MonteCarloSimulation(code, decoder, seed=42)
    
    print(f"\nRunning simulation for {code.get_code_parameters()['notation']} code")
    
    # Estimate logical error rate at specific physical error rate
    physical_error = 0.01
    n_trials = 100  # Increase for more accuracy
    
    print(f"\nTesting physical error rate: {physical_error*100:.1f}%")
    print(f"Number of trials: {n_trials}")
    
    result = sim.estimate_logical_error_rate(
        physical_error_rate=physical_error,
        n_trials=n_trials,
        bias_ratio=1.0
    )
    
    print(f"\nResults:")
    print(f"  Logical error rate: {result['logical_error_rate']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  Failures: {result['n_failures']}/{result['n_trials']}")
    
    print("\n✓ Simulation complete!")


def example_4_threshold():
    """
    Example 4: Find error threshold.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Find Error Threshold")
    print("="*70)
    
    # Setup
    code = HexagonalQutritCode(distance=5)
    decoder = MinimumWeightDecoder(code)
    sim = MonteCarloSimulation(code, decoder, seed=42)
    
    # Define error rates to test
    error_rates = [0.005, 0.01, 0.015]
    n_trials = 100  # Increase for publication
    
    print(f"\nFinding threshold for d={code.distance}")
    print(f"Testing error rates: {[f'{p*100:.1f}%' for p in error_rates]}")
    print(f"Trials per rate: {n_trials}")
    print("\nRunning simulations...\n")
    
    threshold, results = sim.find_threshold(error_rates, n_trials)
    
    print(f"\n{'='*70}")
    print(f"Estimated threshold: {threshold*100:.2f}%")
    print(f"{'='*70}")
    
    print("\nDetailed results:")
    for r in results:
        p_phys = r['physical_error_rate']
        p_log = r['logical_error_rate']
        status = "Below threshold" if p_log < p_phys else "Above threshold"
        print(f"  p={p_phys*100:.1f}% → p_L={p_log:.4f} ({status})")
    
    print("\n✓ Threshold found!")


def example_5_biased_noise():
    """
    Example 5: Test with biased noise.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Biased Noise")
    print("="*70)
    
    # Setup
    code = HexagonalQutritCode(distance=5)
    decoder = MinimumWeightDecoder(code)
    sim = MonteCarloSimulation(code, decoder, seed=42)
    
    physical_error = 0.01
    n_trials = 100
    
    print(f"\nTesting physical error rate: {physical_error*100:.1f}%")
    print(f"Comparing different Z-bias ratios:\n")
    
    for bias in [1, 10, 100]:
        result = sim.estimate_logical_error_rate(
            physical_error_rate=physical_error,
            n_trials=n_trials,
            bias_ratio=bias
        )
        
        print(f"  {bias}:1 bias → p_L = {result['logical_error_rate']:.4f}")
    
    print("\n✓ Biased noise test complete!")
    print("Note: Higher Z-bias typically improves performance")


def main():
    """
    Run all examples.
    """
    print("="*70)
    print("HEXAGONAL QUTRIT ERROR CORRECTION - BASIC EXAMPLES")
    print("="*70)
    print()
    
    try:
        example_1_code_construction()
        example_2_decoder()
        example_3_monte_carlo()
        example_4_threshold()
        example_5_biased_noise()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("  • Run reproduce_paper_results.py for full paper results")
        print("  • Modify these examples for your own experiments")
        print("  • See README.md for more information")
        print()
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
