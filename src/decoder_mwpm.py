"""
Minimum-Weight Perfect Matching Decoder for Hexagonal Qutrit Codes

Implements MWPM algorithm using NetworkX for optimal error correction.

Author: Eddie Chin
Date: November 2025
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple
from itertools import combinations
import matplotlib.pyplot as plt

class MinimumWeightDecoder:
    """
    Minimum-weight perfect matching decoder for qutrit topological codes.
    
    This is the optimal decoder for topological codes under independent errors.
    """
    
    def __init__(self, code):
        """
        Initialize decoder for given code.
        
        Args:
            code: HexagonalQutritCode instance
        """
        self.code = code
        self.lattice = code.lattice
        self.stabilizers = code.stabilizers
        
        # Build decoding graph
        self.decode_graph_x = self._build_decode_graph('X_type')
        self.decode_graph_z = self._build_decode_graph('Z_type')
        
    def _build_decode_graph(self, error_type: str) -> nx.Graph:
        """
        Build decoding graph for minimum-weight matching.
        
        The decoding graph connects syndrome locations with edge weights
        equal to the distance between them in the lattice.
        
        Args:
            error_type: 'X_type' or 'Z_type'
            
        Returns:
            NetworkX graph for decoding
        """
        G_decode = nx.Graph()
        
        # Get stabilizers of this type
        stabilizers = self.stabilizers[error_type]
        positions = self.lattice['positions']
        
        # Each stabilizer becomes a node in decode graph
        # Edge weight = shortest path distance in physical lattice
        for i, stab_i in enumerate(stabilizers):
            for j, stab_j in enumerate(stabilizers):
                if i < j:
                    # Calculate distance between stabilizers
                    # Use center of stabilizer support
                    center_i = np.mean([positions[n] for n in stab_i], axis=0)
                    center_j = np.mean([positions[n] for n in stab_j], axis=0)
                    
                    distance = np.linalg.norm(center_i - center_j)
                    
                    G_decode.add_edge(i, j, weight=distance)
        
        return G_decode
    
    def measure_syndrome(self, errors: Dict[int, Tuple[int, int]]) -> Dict[str, Set[int]]:
        """
        Measure error syndrome (which stabilizers are violated).
        
        Args:
            errors: Dictionary mapping qubit index to (X_error, Z_error) power
                   where X_error, Z_error ∈ {0, 1, 2}
                   
        Returns:
            Dictionary with sets of violated X_type and Z_type stabilizers
        """
        syndrome = {
            'X_type': set(),
            'Z_type': set()
        }
        
        # Check X-type stabilizers (detect Z errors)
        for i, stab in enumerate(self.stabilizers['X_type']):
            # Stabilizer eigenvalue affected by Z errors on its support
            total_z_error = sum(errors.get(qubit, (0, 0))[1] for qubit in stab)
            if total_z_error % 3 != 0:  # Qutrit: mod 3 arithmetic
                syndrome['X_type'].add(i)
        
        # Check Z-type stabilizers (detect X errors)
        for i, stab in enumerate(self.stabilizers['Z_type']):
            # Stabilizer eigenvalue affected by X errors on its support
            total_x_error = sum(errors.get(qubit, (0, 0))[0] for qubit in stab)
            if total_x_error % 3 != 0:
                syndrome['Z_type'].add(i)
        
        return syndrome
    
    def decode(self, syndrome: Dict[str, Set[int]]) -> Dict[int, Tuple[int, int]]:
        """
        Decode syndrome to infer most likely error.
        
        Uses minimum-weight perfect matching on decode graph.
        
        Args:
            syndrome: Dictionary with violated stabilizer indices
            
        Returns:
            Inferred error configuration
        """
        correction = {}
        
        # Decode X errors (from Z-type syndrome)
        if syndrome['Z_type']:
            x_correction = self._decode_single_type(
                syndrome['Z_type'], 
                self.decode_graph_z,
                error_type='X'
            )
            for qubit, power in x_correction.items():
                if qubit not in correction:
                    correction[qubit] = (0, 0)
                correction[qubit] = (power, correction[qubit][1])
        
        # Decode Z errors (from X-type syndrome)
        if syndrome['X_type']:
            z_correction = self._decode_single_type(
                syndrome['X_type'],
                self.decode_graph_x,
                error_type='Z'
            )
            for qubit, power in z_correction.items():
                if qubit not in correction:
                    correction[qubit] = (0, 0)
                correction[qubit] = (correction[qubit][0], power)
        
        return correction
    
    def _decode_single_type(self, syndrome: Set[int], decode_graph: nx.Graph, 
                           error_type: str) -> Dict[int, int]:
        """
        Decode single error type using MWPM.
        
        Args:
            syndrome: Set of violated stabilizer indices
            decode_graph: Decoding graph
            error_type: 'X' or 'Z'
            
        Returns:
            Correction operator as dict of qubit -> error power
        """
        if not syndrome:
            return {}
        
        # Convert syndrome to list
        syndrome_list = list(syndrome)
        
        # If odd number of defects, add boundary node
        if len(syndrome_list) % 2 == 1:
            # Add virtual boundary node at large distance
            boundary_node = max(decode_graph.nodes()) + 1
            for node in syndrome_list:
                decode_graph.add_edge(node, boundary_node, weight=1000.0)
            syndrome_list.append(boundary_node)
        
        # Build subgraph with only syndrome nodes
        syndrome_subgraph = decode_graph.subgraph(syndrome_list).copy()
        
        # Add edges between all syndrome pairs if not present
        for i, j in combinations(syndrome_list, 2):
            if not syndrome_subgraph.has_edge(i, j):
                # Calculate weight as shortest path in full graph
                try:
                    path_length = nx.shortest_path_length(
                        decode_graph, i, j, weight='weight'
                    )
                except nx.NetworkXNoPath:
                    path_length = 1000.0  # Large weight if no path
                
                syndrome_subgraph.add_edge(i, j, weight=path_length)
        
        # Find minimum-weight perfect matching
        matching = nx.min_weight_matching(syndrome_subgraph, weight='weight')
        
        # Convert matching to correction
        correction = {}
        
        for stab_i, stab_j in matching:
            # Skip if involves boundary
            if stab_i >= len(self.stabilizers['X_type']) or \
               stab_j >= len(self.stabilizers['X_type']):
                continue
            
            # Find shortest path between stabilizers in physical lattice
            # This gives us the qubits to apply correction to
            
            # For simplicity, apply correction to midpoint
            # (Full version would trace actual path)
            stab_support_i = self.stabilizers['X_type' if error_type == 'Z' else 'Z_type'][stab_i]
            stab_support_j = self.stabilizers['X_type' if error_type == 'Z' else 'Z_type'][stab_j]
            
            # Apply correction to one qubit from each stabilizer
            qubit_i = list(stab_support_i)[0]
            qubit_j = list(stab_support_j)[0]
            
            # Apply single error (simplified)
            correction[qubit_i] = 1
            correction[qubit_j] = 1
        
        return correction
    
    def calculate_logical_error(self, actual_errors: Dict[int, Tuple[int, int]], 
                                correction: Dict[int, Tuple[int, int]]) -> bool:
        """
        Check if correction produces logical error.
        
        Args:
            actual_errors: Actual errors that occurred
            correction: Decoder's correction
            
        Returns:
            True if logical error occurred (correction failed)
        """
        # Combine actual error and correction
        residual_errors = {}
        
        all_qubits = set(actual_errors.keys()) | set(correction.keys())
        
        for qubit in all_qubits:
            actual = actual_errors.get(qubit, (0, 0))
            corr = correction.get(qubit, (0, 0))
            
            # Residual error (mod 3 for qutrits)
            residual_x = (actual[0] + corr[0]) % 3
            residual_z = (actual[1] + corr[1]) % 3
            
            if residual_x != 0 or residual_z != 0:
                residual_errors[qubit] = (residual_x, residual_z)
        
        # Check if residual error anticommutes with logical operators
        logical_ops = self.code.logical_operators
        
        # Check X logical
        x_logical_support = logical_ops['X_logical']
        total_z_on_x_logical = sum(
            residual_errors.get(q, (0, 0))[1] for q in x_logical_support
        )
        x_logical_error = (total_z_on_x_logical % 3 != 0)
        
        # Check Z logical
        z_logical_support = logical_ops['Z_logical']
        total_x_on_z_logical = sum(
            residual_errors.get(q, (0, 0))[0] for q in z_logical_support
        )
        z_logical_error = (total_x_on_z_logical % 3 != 0)
        
        return x_logical_error or z_logical_error


if __name__ == "__main__":
    from hexagonal_qutrit_code import HexagonalQutritCode
    
    print("="*70)
    print("MINIMUM-WEIGHT PERFECT MATCHING DECODER TEST")
    print("="*70)
    
    # Build code
    code = HexagonalQutritCode(distance=3)
    params = code.get_code_parameters()
    
    print(f"\nCode: {params['notation']}")
    print(f"Physical qutrits: {params['n']}")
    
    # Build decoder
    print("\nBuilding decoder...")
    decoder = MinimumWeightDecoder(code)
    
    print(f"Decode graph for X errors: {decoder.decode_graph_z.number_of_nodes()} nodes")
    print(f"Decode graph for Z errors: {decoder.decode_graph_x.number_of_nodes()} nodes")
    
    # Test with random errors
    print("\n" + "="*70)
    print("Testing decoder with random errors")
    print("="*70)
    
    n_tests = 10
    successes = 0
    
    for test in range(n_tests):
        # Generate random errors (low error rate)
        errors = {}
        error_rate = 0.01
        
        for qubit in range(params['n']):
            if np.random.random() < error_rate:
                # Random X or Z error with random power
                x_power = np.random.randint(0, 3)
                z_power = np.random.randint(0, 3)
                if x_power != 0 or z_power != 0:
                    errors[qubit] = (x_power, z_power)
        
        # Measure syndrome
        syndrome = decoder.measure_syndrome(errors)
        
        # Decode
        correction = decoder.decode(syndrome)
        
        # Check if successful
        logical_error = decoder.calculate_logical_error(errors, correction)
        
        if not logical_error:
            successes += 1
        
        print(f"Test {test+1}: Errors={len(errors)}, "
              f"Syndrome X={len(syndrome['X_type'])}, Z={len(syndrome['Z_type'])}, "
              f"Success={not logical_error}")
    
    success_rate = successes / n_tests
    print(f"\n{'='*70}")
    print(f"Success rate: {success_rate:.1%} ({successes}/{n_tests} tests)")
    print(f"{'='*70}")
