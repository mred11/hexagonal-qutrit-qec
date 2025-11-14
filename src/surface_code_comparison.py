"""
Surface Code Implementation for Comparison

Standard qubit-based surface code for benchmarking against
hexagonal qutrit codes.

Author: Eddie Chin
Date: November 2025
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt

# Import HexagonalQutritCode for comparison functions
try:
    from hexagonal_qutrit_code import HexagonalQutritCode
except ImportError:
    # If running as part of package
    from .hexagonal_qutrit_code import HexagonalQutritCode


class SurfaceCode:
    """
    Implementation of standard surface code (qubit-based) for comparison.
    
    Uses square lattice with data and ancilla qubits.
    """
    
    def __init__(self, distance: int = 3):
        """
        Initialize surface code.
        
        Args:
            distance: Code distance
        """
        self.distance = distance
        self.lattice = self._build_surface_lattice()
        self.stabilizers = self._define_stabilizers()
        
    def _build_surface_lattice(self) -> Dict:
        """
        Build surface code lattice (square lattice with data and ancilla qubits).
        
        Returns:
            Dictionary with lattice structure
        """
        d = self.distance
        
        # Surface code has two types of qubits:
        # - Data qubits: on vertices
        # - Ancilla qubits: on faces (X-type) and vertices (Z-type)
        
        # Simplified: Total qubits ≈ 2 * d^2
        n_data = d * d + (d-1) * (d-1)
        n_ancilla = 2 * d * (d-1)
        n_physical = n_data + n_ancilla
        
        # For comparison purposes, approximate as 2*d^2
        n_physical_approx = 2 * d * d
        
        return {
            'n_data': n_data,
            'n_ancilla': n_ancilla,
            'n_physical': n_physical_approx,
            'n_logical': 1,
            'distance': d
        }
    
    def _define_stabilizers(self) -> Dict:
        """
        Define stabilizer generators.
        
        Returns:
            Dictionary with stabilizer information
        """
        d = self.distance
        
        # Number of stabilizers ≈ 2 * (d-1) * d
        n_x_stabilizers = d * (d-1)
        n_z_stabilizers = d * (d-1)
        
        return {
            'n_X_type': n_x_stabilizers,
            'n_Z_type': n_z_stabilizers
        }
    
    def get_code_parameters(self) -> Dict:
        """
        Get code parameters.
        
        Returns:
            Dictionary with n, k, d
        """
        return {
            'n': self.lattice['n_physical'],
            'k': self.lattice['n_logical'],
            'd': self.distance,
            'notation': f"[[{self.lattice['n_physical']}, {self.lattice['n_logical']}, {self.distance}]]₂"
        }


class SurfaceCodeSimulation:
    """
    Simplified surface code simulation using known thresholds.
    
    Rather than implementing full surface code decoder, we use
    empirically known performance characteristics for comparison.
    """
    
    def __init__(self, distance: int = 3):
        """
        Initialize simulation.
        
        Args:
            distance: Code distance
        """
        self.code = SurfaceCode(distance=distance)
        self.distance = distance
        
        # Known surface code performance (from literature)
        # Threshold ≈ 1.0% for standard depolarizing noise
        # Scales as ~(p/p_th)^((d+1)/2) below threshold
        self.threshold = 0.01
        
    def estimate_logical_error_rate(self, physical_error_rate: float,
                                   bias_ratio: float = 1.0) -> float:
        """
        Estimate logical error rate using scaling formula.
        
        Uses known surface code scaling: p_L ~ (p/p_th)^((d+1)/2)
        
        Args:
            physical_error_rate: Physical error rate
            bias_ratio: Z-bias ratio (surface codes benefit from bias)
            
        Returns:
            Estimated logical error rate
        """
        p = physical_error_rate
        p_th = self.threshold
        d = self.distance
        
        # Adjust threshold for biased noise
        # Surface codes also improve under Z-bias
        if bias_ratio > 1:
            # Empirical: threshold increases roughly as sqrt(bias)
            p_th_biased = p_th * np.sqrt(bias_ratio)
            p_th = min(p_th_biased, 0.03)  # Cap at reasonable value
        
        if p < p_th:
            # Below threshold: exponential suppression
            p_logical = (p / p_th) ** ((d + 1) / 2)
        else:
            # Above threshold: error rate increases
            p_logical = 0.5 * (1 - (1 - 2*p) ** d)
        
        return p_logical
    
    def find_threshold(self, error_rates: List[float],
                      bias_ratio: float = 1.0) -> Tuple[float, List[Dict]]:
        """
        Generate threshold curve for surface code.
        
        Args:
            error_rates: Error rates to test
            bias_ratio: Z-bias ratio
            
        Returns:
            (threshold, results_list)
        """
        results = []
        
        for p_phys in error_rates:
            p_log = self.estimate_logical_error_rate(p_phys, bias_ratio)
            
            results.append({
                'physical_error_rate': p_phys,
                'logical_error_rate': p_log,
                'bias_ratio': bias_ratio
            })
        
        # Threshold
        threshold = self.threshold
        if bias_ratio > 1:
            threshold = threshold * np.sqrt(bias_ratio)
            threshold = min(threshold, 0.03)
        
        return threshold, results


def compare_with_surface_code(hex_results: List[Dict], 
                              surface_results: List[Dict],
                              save_path: str = None):
    """
    Plot comparison between hexagonal qutrit and surface codes.
    
    Args:
        hex_results: Results from hexagonal qutrit simulation
        surface_results: Results from surface code
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hexagonal qutrit data
    hex_p_phys = [r['physical_error_rate'] for r in hex_results]
    hex_p_log = [r['logical_error_rate'] for r in hex_results]
    
    # Surface code data
    surf_p_phys = [r['physical_error_rate'] for r in surface_results]
    surf_p_log = [r['logical_error_rate'] for r in surface_results]
    
    # Plot both
    ax.plot(hex_p_phys, hex_p_log, 'o-', linewidth=2.5, markersize=8,
           label='Hexagonal Qutrit', color='blue')
    ax.plot(surf_p_phys, surf_p_log, 's-', linewidth=2.5, markersize=8,
           label='Surface Code (qubit)', color='red')
    
    # Threshold line
    max_p = max(max(hex_p_phys), max(surf_p_phys))
    ax.plot([0, max_p], [0, max_p], '--', color='black',
           label='Threshold line', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Physical Error Rate', fontsize=13)
    ax.set_ylabel('Logical Error Rate', fontsize=13)
    ax.set_title('Performance Comparison:\nHexagonal Qutrit vs Surface Code',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add text box with key metrics
    if hex_results and surf_results:
        # Find approximate thresholds
        hex_threshold = None
        surf_threshold = 0.01  # Known surface code threshold
        
        for r in hex_results:
            if r['logical_error_rate'] > r['physical_error_rate']:
                hex_threshold = r['physical_error_rate']
                break
        
        if hex_threshold:
            textstr = f'Thresholds:\n'
            textstr += f'Hex Qutrit: ~{hex_threshold*100:.2f}%\n'
            textstr += f'Surface Code: ~{surf_threshold*100:.2f}%'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
                   fontsize=11, verticalalignment='bottom',
                   horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    return fig


def compare_resource_efficiency(distances: List[int], save_path: str = None):
    """
    Compare resource requirements (qubit count) for same code distance.
    
    Args:
        distances: List of code distances to compare
        save_path: Optional save path
    """
    hex_qubits = []
    surf_qubits = []
    
    for d in distances:
        # Hexagonal qutrit
        hex_code = HexagonalQutritCode(distance=d)
        n_qutrits = hex_code.get_code_parameters()['n']
        # Convert to qubit equivalents (log2(3) ≈ 1.585 bits per qutrit)
        n_hex_qubit_equiv = n_qutrits * np.log2(3)
        hex_qubits.append(n_hex_qubit_equiv)
        
        # Surface code
        surf_code = SurfaceCode(distance=d)
        n_surf_qubits = surf_code.get_code_parameters()['n']
        surf_qubits.append(n_surf_qubits)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.array(distances)
    
    ax.plot(x, hex_qubits, 'o-', linewidth=2.5, markersize=10,
           label='Hexagonal Qutrit (qubit-equivalent)', color='blue')
    ax.plot(x, surf_qubits, 's-', linewidth=2.5, markersize=10,
           label='Surface Code (qubits)', color='red')
    
    # Calculate savings
    savings = [(s - h) / s * 100 for h, s in zip(hex_qubits, surf_qubits)]
    
    ax.set_xlabel('Code Distance', fontsize=13)
    ax.set_ylabel('Physical Qubits / Qubit-Equivalents', fontsize=13)
    ax.set_title('Resource Efficiency Comparison',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add savings annotation
    for i, d in enumerate(distances):
        if i < len(savings):
            ax.annotate(f'{savings[i]:.1f}% savings',
                       xy=(d, hex_qubits[i]),
                       xytext=(10, -20), textcoords='offset points',
                       fontsize=9, color='green',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Resource comparison saved to {save_path}")
    
    return fig, savings


if __name__ == "__main__":
    from hexagonal_qutrit_code import HexagonalQutritCode
    
    print("="*70)
    print("SURFACE CODE COMPARISON")
    print("="*70)
    
    distance = 5
    
    # Surface code parameters
    surf_code = SurfaceCode(distance=distance)
    surf_params = surf_code.get_code_parameters()
    
    print(f"\nSurface Code: {surf_params['notation']}")
    print(f"Physical qubits: {surf_params['n']}")
    
    # Hexagonal qutrit parameters
    hex_code = HexagonalQutritCode(distance=distance)
    hex_params = hex_code.get_code_parameters()
    
    print(f"\nHexagonal Qutrit: {hex_params['notation']}")
    print(f"Physical qutrits: {hex_params['n']}")
    print(f"Qubit-equivalents: {hex_params['n'] * np.log2(3):.1f}")
    
    # Resource savings
    savings = (surf_params['n'] - hex_params['n'] * np.log2(3)) / surf_params['n'] * 100
    print(f"\nResource savings: {savings:.1f}%")
    
    # Compare resource scaling
    print(f"\n{'='*70}")
    print("RESOURCE SCALING COMPARISON")
    print(f"{'='*70}")
    
    distances = [3, 5, 7]
    fig, savings_list = compare_resource_efficiency(
        distances,
        save_path='/mnt/user-data/outputs/resource_comparison.png'
    )
    
    print("\nResource comparison complete!")
