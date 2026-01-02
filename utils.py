"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          Graph management and visualization utilities for RQAOA
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides:
  • A GraphManager class for Recursive QAOA (RQAOA), implementing in-place
    correlation and anti-correlation variable eliminations.
  • Bookkeeping utilities to track node mappings, eliminated variables, and
    spin-sign propagation required to reconstruct full assignments.
  • An exact brute-force solver for the small residual instance after RQAOA
    eliminations.
  • Visualization helpers for debugging intermediate reduced graphs, including
    support for signed edge weights and optional external fields.

Implementation Notes
--------------------
• Graph mutation:
  The reduced graph is mutated in-place during elimination. All changes are
  logged, and node mappings are tracked to ensure consistent reconstruction
  of assignments for the original problem.

• External fields:
  If `fields_present=True`, diagonal (node) weights are treated as external
  fields in the Ising Hamiltonian and propagated correctly during elimination.

• Performance:
  This module prioritizes correctness and transparency over speed. The
  elimination routines are not JIT-compiled and are intended for small to
  medium problem sizes typical of the RQAOA reduction stage.

How to Cite
-----------
If you use this code in academic work, please cite:
  Classical and Quantum Heuristics for the Binary Paint Shop Problem,
  https://arxiv.org/abs/2509.15294

License
-------
MIT License © 2026 V. Vijendran

==============================================================================
"""


# -------------------------
# Standard library imports
# -------------------------
import sys
import itertools
from functools import partial  # (kept if used elsewhere)

# -------------------------
# Third-party imports
# -------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plotting helpers (debugging)
# ---------------------------------------------------------------------------

def draw_graph(G: nx.Graph) -> None:
    """
    Plot a weighted NetworkX graph using a circular layout.

    This helper is primarily intended for debugging RQAOA eliminations:
    it draws positive-weight edges as solid lines and negative-weight edges
    as dashed lines.

    Parameters
    ----------
    G : nx.Graph
        Weighted graph with edge attribute 'weight'.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=250)

    epositive = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("weight", 0.0) >= 0.0]
    enegative = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("weight", 0.0) < 0.0]

    nx.draw_networkx_edges(G, pos, edgelist=epositive, width=3)
    nx.draw_networkx_edges(
        G, pos, edgelist=enegative,
        width=3, alpha=0.5, edge_color="b", style="dashed"
    )

    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

    plt.show()


def draw_graph_with_fields(G: nx.Graph) -> None:
    """
    Plot a weighted NetworkX graph including node-field values (if present).

    Edges are styled the same way as `draw_graph`. In addition, node attributes
    named 'weight' (external fields) are drawn as red labels near each node.

    Parameters
    ----------
    G : nx.Graph
        Weighted graph with edge attribute 'weight' and optional node attribute
        'weight' for external fields.
    """
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")

    epositive = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("weight", 0.0) >= 0.0]
    enegative = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("weight", 0.0) < 0.0]

    nx.draw_networkx_edges(G, pos, edgelist=epositive, width=2)
    nx.draw_networkx_edges(
        G, pos, edgelist=enegative,
        width=2, alpha=0.5, edge_color="b", style="dashed"
    )

    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

    # Offset red node-field labels slightly for readability
    label_offset = 0.05
    pos_labels = {node: (xy[0] + label_offset, xy[1] + label_offset) for node, xy in pos.items()}

    node_labels = nx.get_node_attributes(G, "weight")
    if node_labels:
        nx.draw_networkx_labels(G, pos_labels, labels=node_labels, font_color="red")

    plt.show()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def has_edge(edge: tuple[int, int], edge_list: list[tuple[int, int]]) -> bool:
    """
    Check whether an undirected edge appears in an edge list.

    Parameters
    ----------
    edge : (int, int)
        Candidate edge (u, v).
    edge_list : list[(int, int)]
        List of edges.

    Returns
    -------
    bool
        True iff (u, v) or (v, u) is in edge_list.
    """
    return (edge in edge_list) or (edge[::-1] in edge_list)


def graph_to_array(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a NetworkX graph to (edges, adjacency matrix).

    Parameters
    ----------
    graph : nx.Graph
        Input graph.

    Returns
    -------
    edges : np.ndarray, shape (m, 2)
        Edge list as integer node pairs.
    adj_mat : np.ndarray, shape (n, n)
        Weighted adjacency matrix consistent with sorted node ordering.
    """
    node_list = sorted(graph.nodes())
    adj_mat = nx.to_numpy_array(graph, nodelist=node_list)

    # Return edges in a reproducible order (useful for debugging/repro)
    edges = np.array(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=int)
    return edges, adj_mat


# ---------------------------------------------------------------------------
# GraphManager for RQAOA eliminations
# ---------------------------------------------------------------------------

class GraphManager:
    """
    Stateful manager for RQAOA variable elimination and solution reconstruction.

    Responsibilities
    ----------------
    1) Store the original problem graph (for final objective evaluation).
    2) Maintain a reduced graph that is mutated by elimination steps:
       • correlate(u, v): enforce u =  v and eliminate u
       • anti_correlate(u, v): enforce u = -v and eliminate u
    3) Track elimination mappings in `node_maps` so that assignments on the
       residual graph can be propagated back to the original variables.

    Attributes
    ----------
    original_graph : nx.Graph
        Unmodified copy of the input graph.
    reduced_graph : nx.Graph
        Graph that is mutated in-place during elimination.
    fields_present : bool
        Whether node 'weight' attributes (external fields) are used.
    nodes_vals : dict[int, int]
        Final spin assignments in {+1, -1} for all original nodes.
    node_maps : dict[int, (int, int)]
        Mapping: eliminated_node -> (mapped_to_node, sign), where sign ∈ {+1, -1}.
    remaining_nodes : list[int]
        Nodes not eliminated yet (defines the residual brute-force space).
    log : dict[int, str]
        Human-readable trace of operations.
    """

    def __init__(self, graph: nx.Graph, fields_present: bool = False, verbose: bool = False):
        self.original_graph = graph.copy()
        self.reduced_graph = graph  # NOTE: this mutates the passed graph in-place
        self.verbose = verbose
        self.fields_present = fields_present

        n = graph.number_of_nodes()
        self.nodes_vals = {i: 0 for i in range(n)}
        self.node_maps = {i: (i, 1) for i in range(n)}
        self.remaining_nodes = [i for i in range(n)]

        self.iter = 0
        self.optimal_angles = {}
        self.log = {}

    def correlate(self, edge: tuple[int, int]) -> None:
        """
        Eliminate node u by enforcing u = v (positive correlation) on edge (u, v).

        This removes u from the reduced graph and reattaches its incident edges
        onto v (summing weights when an edge already exists). If fields are present,
        node weights are also merged.

        Parameters
        ----------
        edge : (int, int)
            Edge (u, v) in the reduced graph.
        """
        u, v = edge
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."

        # Neighbors of u excluding v
        d = list(self.reduced_graph[u])
        d.remove(v)

        # Record mapping u -> v with +1 sign
        self.node_maps[u] = (v, 1)
        self.remaining_nodes.remove(u)

        self._log(f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph.")
        self.reduced_graph.remove_edge(v, u)

        old_weights = {w: self.reduced_graph[w][u]["weight"] for w in d}

        # Remove u's incident edges
        for w in d:
            self._log(f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph.")
            self.reduced_graph.remove_edge(w, u)

        # Merge fields if present, then remove u
        self._log(f"Removing node {u} from graph.")
        if self.fields_present:
            self.reduced_graph.nodes[v]["weight"] += self.reduced_graph.nodes[u]["weight"]
        self.reduced_graph.remove_node(u)

        # Reattach edges (w, v) with merged weights
        new_edges = {(w, v): old_weights[w] for w in d}
        for (w, vv) in list(new_edges.keys()):
            if self.reduced_graph.has_edge(w, vv):
                new_edges[(w, vv)] += self.reduced_graph[w][vv]["weight"]

        for (w, vv), weight in new_edges.items():
            if weight == 0.0:
                self._log(f"Removing edge ({w}, {vv}) with weight {weight} from graph.")
                if self.reduced_graph.has_edge(w, vv):
                    self.reduced_graph.remove_edge(w, vv)
            else:
                self._log(f"Adding edge ({w}, {vv}) with weight {weight} to graph.")
                self.reduced_graph.add_edge(w, vv, weight=weight)

    def anti_correlate(self, edge: tuple[int, int]) -> None:
        """
        Eliminate node u by enforcing u = -v (negative correlation) on edge (u, v).

        Identical to `correlate` except that incident edge weights from u are
        reattached with a sign flip (and node fields are merged with a minus sign).

        Parameters
        ----------
        edge : (int, int)
            Edge (u, v) in the reduced graph.
        """
        u, v = edge
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."

        d = list(self.reduced_graph[u])
        d.remove(v)

        self.node_maps[u] = (v, -1)
        self.remaining_nodes.remove(u)

        self._log(f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph.")
        self.reduced_graph.remove_edge(v, u)

        old_weights = {w: self.reduced_graph[w][u]["weight"] for w in d}

        for w in d:
            self._log(f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph.")
            self.reduced_graph.remove_edge(w, u)

        self._log(f"Removing node {u} from graph.")
        if self.fields_present:
            self.reduced_graph.nodes[v]["weight"] += -1.0 * self.reduced_graph.nodes[u]["weight"]
        self.reduced_graph.remove_node(u)

        # Sign-flip when reattaching edges
        new_edges = {(w, v): -old_weights[w] for w in d}
        for (w, vv) in list(new_edges.keys()):
            if self.reduced_graph.has_edge(w, vv):
                new_edges[(w, vv)] += self.reduced_graph[w][vv]["weight"]

        for (w, vv), weight in new_edges.items():
            if weight == 0.0:
                self._log(f"Removing edge ({w}, {vv}) with weight {weight} from graph.")
                if self.reduced_graph.has_edge(w, vv):
                    self.reduced_graph.remove_edge(w, vv)
            else:
                self._log(f"Adding edge ({w}, {vv}) with weight {weight} to graph.")
                self.reduced_graph.add_edge(w, vv, weight=weight)

    def eliminate_node(self, node: int, sign: int) -> None:
        """
        Eliminate a node directly by absorbing its incident couplings into neighbors.

        This is used when external fields are present (diagonal terms), where
        eliminating a node updates neighbor fields. The assignment for the node
        is recorded in `nodes_vals`.

        Parameters
        ----------
        node : int
            Node to eliminate.
        sign : int
            Spin assignment (+1 or -1) to apply.
        """
        neighbours = list(self.reduced_graph[node])

        for neighbour in neighbours:
            self.reduced_graph.nodes[neighbour]["weight"] += sign * self.reduced_graph[node][neighbour]["weight"]
            self._log(f"Removing edge ({node}, {neighbour}) with weight {self.reduced_graph[node][neighbour]['weight']} from graph.")
            self.reduced_graph.remove_edge(node, neighbour)

        self._log(f"Removing node {node} with weight {self.reduced_graph.nodes[node].get('weight', 0.0)} from graph.")

        self.nodes_vals[node] = sign
        self.remaining_nodes.remove(node)
        self.reduced_graph.remove_node(node)

    def get_root_node(self, node: int, s: int) -> tuple[int, int]:
        """
        Follow chained node_maps until reaching a root (non-eliminated) node.

        Parameters
        ----------
        node : int
            Starting (possibly eliminated) node.
        s : int
            Accumulated sign (+1 or -1) from previous mappings.

        Returns
        -------
        (root_node, sign) : (int, int)
            root_node is a remaining node (or a fixed-assigned node if fields_present),
            sign is the product of correlation signs along the mapping chain.
        """
        mapped_node, sign = self.node_maps[node]
        sign *= s

        if self.fields_present:
            # Root if still remaining OR already assigned via eliminate_node
            if (mapped_node in self.remaining_nodes) or (self.nodes_vals[mapped_node] != 0):
                return mapped_node, sign
            return self.get_root_node(mapped_node, sign)

        if mapped_node in self.remaining_nodes:
            return mapped_node, sign
        return self.get_root_node(mapped_node, sign)

    def set_node_values(self, values: list[int]) -> None:
        """
        Assign spins to remaining nodes and propagate assignments to eliminated nodes.

        Parameters
        ----------
        values : list[int]
            Spin assignments for `remaining_nodes`, entries must be in {+1, -1}.
        """
        assert len(values) == len(self.remaining_nodes), "Values length must match remaining_nodes length."
        for v in values:
            assert v in (1, -1), "Values must be +1 or -1."

        # Assign remaining nodes
        for node, value in zip(self.remaining_nodes, values):
            self.nodes_vals[node] = value

        # Propagate to eliminated nodes
        for node, (mapped_node, sign) in self.node_maps.items():
            if node == mapped_node:
                continue  # root maps to itself
            if mapped_node in self.remaining_nodes:
                self.nodes_vals[node] = sign * self.nodes_vals[mapped_node]
            else:
                root_node, s = self.get_root_node(mapped_node, sign)
                self.nodes_vals[node] = s * self.nodes_vals[root_node]

    def compute_cost(self, graph: nx.Graph) -> float:
        """
        Compute the Ising / Max-Cut-style energy for the given graph.

        Cost convention:
          sum_{(i,j) in E} w_ij * s_i * s_j  (+ optional node fields)

        Parameters
        ----------
        graph : nx.Graph
            Graph on which to evaluate the objective.

        Returns
        -------
        float
            Total energy under the current `nodes_vals`.
        """
        for value in self.nodes_vals.values():
            assert value in (1, -1), "All nodes must have value +1 or -1."

        total_cost = 0.0
        for u, v in graph.edges():
            total_cost += graph[u][v]["weight"] * self.nodes_vals[u] * self.nodes_vals[v]

        if self.fields_present:
            for node in graph.nodes():
                total_cost += graph.nodes[node]["weight"] * self.nodes_vals[node]

        return total_cost

    def brute_force(self):
        """
        Brute-force the residual graph over remaining_nodes and return best assignment.

        This enumerates all 2^(|remaining_nodes|) assignments, so it is only intended
        for the *small* residual instance after eliminations.

        Returns
        -------
        (best_cost, nodes_vals) : (float, dict[int, int])
            best_cost is computed on the original graph; nodes_vals contains the
            reconstructed assignment for all original nodes.
        """
        num_values = len(self.remaining_nodes)
        assignments = itertools.product([1, -1], repeat=num_values)

        best_reduced_cost = sys.maxsize
        best_assignment = None

        for assignment in assignments:
            assignment = list(assignment)
            self.set_node_values(assignment)
            reduced_cost = self.compute_cost(self.reduced_graph)

            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_assignment = assignment
                self._log(f"Best reduced cost so far: {best_reduced_cost} for assignment {assignment}.")

        self.set_node_values(best_assignment)
        best_cost = self.compute_cost(self.original_graph)

        self._log(f"Best cost on original graph: {best_cost}.")
        return best_cost, self.nodes_vals

    def _log(self, msg: str) -> None:
        """Internal helper: append msg to current iteration log and optionally print."""
        if self.iter not in self.log:
            self.log[self.iter] = ""
        self.log[self.iter] += msg + "\n"
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Data extraction for JIT kernels
# ---------------------------------------------------------------------------

def extract_properties(graphmanager: GraphManager) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (edges, adjacency matrix) from the current reduced graph.

    The adjacency matrix is embedded into the original problem size so that
    node indices remain consistent across eliminations.

    Parameters
    ----------
    graphmanager : GraphManager
        Current manager holding reduced_graph and original_graph.

    Returns
    -------
    red_edges : np.ndarray, shape (m, 2)
        Edge list for the reduced instance.
    adj_mat : np.ndarray, shape (N, N)
        Dense weight matrix in the original indexing convention. For edges
        present in reduced_graph, adj_mat[u, v] = w_uv. If fields are present,
        diagonal entries adj_mat[i, i] store node field weights.
    """
    red_edges, _ = graph_to_array(graphmanager.reduced_graph)

    num_nodes = graphmanager.original_graph.number_of_nodes()
    adj_mat = np.zeros((num_nodes, num_nodes), dtype=float)

    for u, v in red_edges:
        w = graphmanager.reduced_graph[u][v]["weight"]
        adj_mat[u, v] = w
        adj_mat[v, u] = w

    if graphmanager.fields_present:
        for i in graphmanager.reduced_graph.nodes:
            adj_mat[i, i] = graphmanager.reduced_graph.nodes[i]["weight"]

    return red_edges, adj_mat
