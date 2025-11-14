"""
Hexagonal Qutrit Quantum Error Correction Code
Implementation of stabilizer-based topological codes on hexagonal lattices

Author: Eddie Chin
Date: November 2025
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import matplotlib.pyplot as plt

class HexagonalQutritCode:
    """
    Implementation of quantum error correction code on hexagonal lattice with qutrits.
    
    Based on stabilizer formalism extended to d=3 (qutrit) systems.
    """
    
    def __init__(self, distance: int = 3):
        """
        Initialize hexagonal qutrit code.
        
        Args:
            distance: Code distance (larger = better protection, more qubits)
        """
        self.distance = distance
        self.lattice = self._build_hexagonal_lattice()
        self.stabilizers = self._define_stabilizers()
        self.logical_operators = self._define_logical_operators()
        
    def _build_hexagonal_lattice(self) -> Dict:
        """
        Build hexagonal lattice structure (honeycomb pattern).
        
        Returns:
            Dictionary containing graph, positions, and code parameters
        """
        G = nx.Graph()
        positions = {}
        
        # Build hexagonal grid using offset coordinates
        node_id = 0
        d = self.distance
        
        for row in range(-d, d + 1):
            for col in range(-d, d + 1):
                # Hexagonal coordinate system
                x = col * 1.5
                y = row * np.sqrt(3) + (col % 2) * np.sqrt(3) / 2
                
                # Keep nodes within distance bound
                if abs(row) + abs(col) + abs(row + col) <= 2 * d:
                    positions[node_id] = np.array([x, y])
                    G.add_node(node_id, state=0)  # Initialize in |0⟩ state
                    node_id += 1
        
        # Connect nearest neighbors (hexagonal has 6 neighbors per node)
        nodes_list = list(G.nodes())
        for i in nodes_list:
            for j in nodes_list:
                if i < j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    # Nearest neighbor threshold (allow some numerical tolerance)
                    if 1.4 < dist < 1.8:
                        G.add_edge(i, j)
        
        n_physical = G.number_of_nodes()
        n_logical = 1  # Encode 1 logical qutrit
        
        return {
            'graph': G,
            'positions': positions,
            'n_physical': n_physical,
            'n_logical': n_logical,
            'distance': self.distance,
            'coordination': 6  # Hexagonal coordination number
        }
    
    def _define_stabilizers(self) -> Dict[str, List[Set[int]]]:
        """
        Define stabilizer generators for the code.
        
        For qutrits, we have generalized Pauli operators X and Z where:
        X = |0⟩⟨1| + |1⟩⟨2| + |2⟩⟨0| (cyclic shift)
        Z = |0⟩⟨0| + ω|1⟩⟨1| + ω²|2⟩⟨2| (phase, ω = e^(2πi/3))
        
        Returns:
            Dictionary with 'X_type' and 'Z_type' stabilizer generators
        """
        G = self.lattice['graph']
        
        # X-type stabilizers: detect Z errors
        # Applied to faces (plaquettes) of hexagonal lattice
        x_stabilizers = []
        
        # Find all hexagonal faces
        faces = self._find_hexagonal_faces(G)
        for face in faces:
            x_stabilizers.append(set(face))
        
        # Z-type stabilizers: detect X errors
        # Applied to vertices (stars)
        z_stabilizers = []
        
        for node in G.nodes():
            # Get neighbors
            neighbors = list(G.neighbors(node))
            if len(neighbors) >= 3:
                # For irregular boundaries, use subset of neighbors
                z_stabilizers.append(set([node] + neighbors[:3]))
        
        return {
            'X_type': x_stabilizers,
            'Z_type': z_stabilizers
        }
    
    def _find_hexagonal_faces(self, G: nx.Graph) -> List[List[int]]:
        """
        Find all hexagonal faces (6-cycles) in the lattice.
        
        Args:
            G: NetworkX graph of the lattice
            
        Returns:
            List of node lists forming hexagonal faces
        """
        faces = []
        visited_faces = set()
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            
            # Look for 6-cycles containing this node
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    # Check if n1 and n2 are connected
                    if G.has_edge(n1, n2):
                        # Try to complete the hexagon
                        face = self._try_complete_hexagon(G, node, n1, n2)
                        if face and len(face) == 6:
                            # Canonicalize face (sorted tuple for uniqueness)
                            face_key = tuple(sorted(face))
                            if face_key not in visited_faces:
                                faces.append(face)
                                visited_faces.add(face_key)
        
        return faces
    
    def _try_complete_hexagon(self, G: nx.Graph, start: int, n1: int, n2: int) -> List[int]:
        """
        Try to complete a hexagon starting from three nodes.
        
        Args:
            G: Graph
            start: Starting node
            n1, n2: Two neighbors of start
            
        Returns:
            List of 6 nodes forming hexagon, or None if not found
        """
        # Simple BFS to find 6-cycle
        # This is a simplified version; full implementation would be more robust
        
        # Get common neighbors that could complete hexagon
        n1_neighbors = set(G.neighbors(n1)) - {start}
        n2_neighbors = set(G.neighbors(n2)) - {start}
        
        for n3 in n1_neighbors:
            if n3 in n2_neighbors:
                continue
            for n4 in n2_neighbors:
                if G.has_edge(n3, n4):
                    # Check if we can close the hexagon
                    n3_neighbors = set(G.neighbors(n3)) - {n1, n4}
                    n4_neighbors = set(G.neighbors(n4)) - {n2, n3}
                    common = n3_neighbors & n4_neighbors
                    if common and G.has_edge(list(common)[0], start):
                        return [start, n1, n3, list(common)[0], n4, n2]
        
        return None
    
    def _define_logical_operators(self) -> Dict[str, List[int]]:
        """
        Define logical X and Z operators for the encoded qutrit.
        
        Returns:
            Dictionary with 'X_logical' and 'Z_logical' operator support
        """
        G = self.lattice['graph']
        positions = self.lattice['positions']
        
        # Logical operators are non-contractible loops through the lattice
        # X_logical: horizontal path
        # Z_logical: vertical path
        
        # Find leftmost and rightmost nodes
        x_coords = [pos[0] for pos in positions.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        
        # Logical X: path from left to right
        left_nodes = [n for n, pos in positions.items() if abs(pos[0] - min_x) < 0.5]
        right_nodes = [n for n, pos in positions.items() if abs(pos[0] - max_x) < 0.5]
        
        # Find shortest path (simplified)
        try:
            x_logical = nx.shortest_path(G, left_nodes[0], right_nodes[0])
        except:
            x_logical = []
        
        # Logical Z: orthogonal path
        y_coords = [pos[1] for pos in positions.values()]
        min_y, max_y = min(y_coords), max(y_coords)
        
        bottom_nodes = [n for n, pos in positions.items() if abs(pos[1] - min_y) < 0.5]
        top_nodes = [n for n, pos in positions.items() if abs(pos[1] - max_y) < 0.5]
        
        try:
            z_logical = nx.shortest_path(G, bottom_nodes[0], top_nodes[0])
        except:
            z_logical = []
        
        return {
            'X_logical': x_logical,
            'Z_logical': z_logical
        }
    
    def get_code_parameters(self) -> Dict:
        """
        Get code parameters in standard notation [[n, k, d]].
        
        Returns:
            Dictionary with code parameters
        """
        n = self.lattice['n_physical']
        k = self.lattice['n_logical']
        d = self.distance
        
        return {
            'n': n,  # Physical qutrits
            'k': k,  # Logical qutrits
            'd': d,  # Code distance
            'notation': f'[[{n}, {k}, {d}]]₃'  # Standard notation
        }
    
    def visualize_lattice(self, save_path: str = None):
        """
        Visualize the hexagonal lattice structure.
        
        Args:
            save_path: Optional path to save figure
        """
        G = self.lattice['graph']
        pos = self.lattice['positions']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw lattice
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        # Highlight stabilizers
        x_stabs = self.stabilizers['X_type']
        z_stabs = self.stabilizers['Z_type']
        
        # Draw a few example stabilizers
        if x_stabs:
            example_x = list(x_stabs[0])
            subgraph_x = G.subgraph(example_x)
            nx.draw_networkx_nodes(subgraph_x, pos, node_color='red',
                                  node_size=400, alpha=0.5, ax=ax)
        
        if z_stabs:
            example_z = list(z_stabs[0])
            subgraph_z = G.subgraph(example_z)
            nx.draw_networkx_nodes(subgraph_z, pos, node_color='green',
                                  node_size=350, alpha=0.5, ax=ax)
        
        ax.set_title(f'Hexagonal Qutrit Code (distance={self.distance})\n'
                    f'{self.get_code_parameters()["notation"]}',
                    fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Lattice visualization saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("HEXAGONAL QUTRIT QUANTUM ERROR CORRECTION CODE")
    print("="*70)
    
    for distance in [3, 5, 7]:
        print(f"\n{'='*70}")
        print(f"Code Distance: {distance}")
        print(f"{'='*70}")
        
        code = HexagonalQutritCode(distance=distance)
        params = code.get_code_parameters()
        
        print(f"\nCode Parameters: {params['notation']}")
        print(f"Physical qutrits: {params['n']}")
        print(f"Logical qutrits: {params['k']}")
        print(f"Code distance: {params['d']}")
        print(f"Coordination number: {code.lattice['coordination']}")
        
        print(f"\nStabilizers:")
        print(f"  X-type (detect Z errors): {len(code.stabilizers['X_type'])}")
        print(f"  Z-type (detect X errors): {len(code.stabilizers['Z_type'])}")
        
        print(f"\nLogical operators:")
        print(f"  X_logical support: {len(code.logical_operators['X_logical'])} qubits")
        print(f"  Z_logical support: {len(code.logical_operators['Z_logical'])} qubits")
        
        # Visualize d=5 case
        if distance == 5:
            code.visualize_lattice(f'/mnt/user-data/outputs/hexagonal_lattice_d{distance}.png')
    
    print(f"\n{'='*70}")
    print("Code construction complete!")
    print(f"{'='*70}")
