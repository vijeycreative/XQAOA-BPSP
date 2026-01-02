"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          BPSP instance generation and Ising/graph mapping utilities
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides small, dependency-light utilities used throughout the
Binary Paint Shop Problem (BPSP) experiments:

  • Random BPSP instance generation via Fisher–Yates shuffling.
  • Conversion of a BPSP car sequence into a weighted Ising / Max-Cut graph
    (NetworkX Graph) using the standard adjacent-pair coupling rule.
  • Helpers to extract (edges, adjacency-matrix) representations used by
    NumPy/Numba/JAX solvers.
  • Conversion from a binary (or ±1) color assignment into a paint sequence
    and its number of paint swaps.

Implementation Notes
--------------------
• Car indexing:
  Most routines assume cars are labeled 0..n-1 and appear exactly twice in the
  car sequence. `generate_bpsp` constructs sequences of length 2n accordingly.

• Graph representation:
  `map_ising` internally accumulates edge weights in a dict keyed by (u, v)
  with u < v. The final output is a NetworkX undirected graph with edge
  attribute name `"weight"`.

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

from __future__ import annotations

import numpy as np
import networkx as nx


# -----------------------------------------------------------------------------
# Random instance generation
# -----------------------------------------------------------------------------

def shuffle(arr: np.ndarray) -> np.ndarray:
    """
    In-place Fisher–Yates shuffle for a 1D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Array to be shuffled. The shuffle is performed *in-place*.

    Returns
    -------
    np.ndarray
        The same array object (shuffled), returned for convenience.

    Notes
    -----
    • This is equivalent to `np.random.shuffle(arr)` but kept explicit for
      reproducibility/clarity in the paper code.
    • Uses NumPy's global RNG (`np.random.randint`). If you want deterministic
      experiments, set a seed externally via `np.random.seed(seed)`.
    """
    n = len(arr)
    # Walk backwards, swapping each position with a random earlier index.
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def generate_bpsp(n: int) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """
    Generate a random BPSP instance with n cars.

    The output sequence has length 2n, containing each car label in {0,...,n-1}
    exactly twice. The sequence is uniformly randomized (Fisher–Yates).

    Parameters
    ----------
    n : int
        Number of distinct cars.

    Returns
    -------
    sequence : np.ndarray, shape (2n,)
        Randomized car sequence containing each label twice.
    car_pos : dict[int, tuple[int, int]]
        Mapping car -> (first_position, second_position) in the sequence.
    """
    # Build [0,1,...,n-1, 0,1,...,n-1] and shuffle.
    sequence = shuffle(np.array(list(range(n)) + list(range(n))))
    car_pos = get_pos(sequence)
    return sequence, car_pos


def get_pos(sequence: np.ndarray) -> dict[int, tuple[int, int]]:
    """
    Compute (first, second) occurrence positions for each car label in a sequence.

    Assumes the sequence has even length 2n and each label 0..n-1 appears twice.

    Parameters
    ----------
    sequence : np.ndarray
        Sequence of car labels.

    Returns
    -------
    dict[int, tuple[int, int]]
        car -> (pos_a, pos_b)

    Raises
    ------
    ValueError
        If a label does not appear exactly twice.
    """
    n = len(sequence) // 2
    car_pos: dict[int, tuple[int, int]] = {i: (-1, -1) for i in range(n)}

    for i in range(n):
        idx = np.where(sequence == i)[0]
        if idx.shape[0] != 2:
            raise ValueError(f"Car label {i} appears {idx.shape[0]} times; expected exactly 2.")
        car_pos[i] = (int(idx[0]), int(idx[1]))

    return car_pos


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------

def nx_graph(g: dict[tuple[int, int], float]) -> nx.Graph:
    """
    Convert an edge-weight dictionary into a NetworkX weighted undirected graph.

    Parameters
    ----------
    g : dict[(u, v), w]
        Edge dictionary. Keys are node pairs; values are weights. Zero-weight
        entries are ignored.

    Returns
    -------
    nx.Graph
        Undirected graph with edge attribute `"weight"`.
    """
    weighted_edges = []
    for (u, v), w in g.items():
        if w != 0:
            weighted_edges.append((u, v, w))

    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)  # stores weights under attribute "weight"
    return G


def map_ising(car_sequence: np.ndarray) -> nx.Graph:
    """
    Map a BPSP car sequence into a weighted Ising/Max-Cut graph.

    Construction
    -----------
    We scan adjacent pairs (car_sequence[i], car_sequence[i+1]) from left to right.
    Let `car_occr[k]` be the number of times car k has appeared *so far* while scanning.
    For an adjacent pair (a, b), define:
        weight = (-1)^(car_occr[a] + car_occr[b] + 1)
    and add this coupling to the edge (a, b) in an undirected weighted graph.
    Multiple contributions to the same edge are summed.

    Parameters
    ----------
    car_sequence : np.ndarray, shape (2n,)
        Sequence of car labels 0..n-1, each appearing twice.

    Returns
    -------
    nx.Graph
        Weighted undirected graph representing the Ising instance.
        Nodes with no incident edges are still included.

    Notes
    -----
    • This routine assumes cars are labeled 0..n-1 (contiguous labels).
    • Any self-loop terms (u == v) are removed and accumulated into a constant.
      (The constant is not returned here because downstream solvers typically
      work with the graph-only part; reintroduce it if your pipeline needs it.)
    """
    num_cars = len(car_sequence) // 2

    # car_occr[k] tracks how many times we've seen car k so far in the left-to-right scan.
    car_occr = [0 for _ in range(num_cars)]

    # Accumulate weights in a dict keyed by (min(u,v), max(u,v)).
    graph: dict[tuple[int, int], int] = {}

    def add_edge(u: int, v: int, w: int) -> None:
        # Normalize key ordering for undirected edges.
        edge = (u, v) if u < v else (v, u)
        if edge in graph:
            graph[edge] += w
        else:
            graph[edge] = w

    # Scan adjacent pairs in the sequence.
    for i in range(len(car_sequence) - 1):
        a = int(car_sequence[i])
        b = int(car_sequence[i + 1])

        # Number of prior occurrences determines the sign pattern.
        occ_a = car_occr[a]
        occ_b = car_occr[b]
        w = (-1) ** (occ_a + occ_b + 1)

        add_edge(a, b, w)

        # Increment only the left car's occurrence count, matching the original convention.
        car_occr[a] += 1

    # Remove any accidental self-loops (rare; depends on conventions/data).
    constant = 0
    for (u, v) in list(graph.keys()):
        if u == v:
            constant += graph[(u, v)]
            del graph[(u, v)]

    # Build NetworkX graph and ensure all cars appear as nodes.
    ising_graph = nx_graph(graph)
    for i in range(num_cars):
        if i not in ising_graph:
            ising_graph.add_node(i)

    return ising_graph


def get_edges_adj_mat(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a sorted edge list and dense adjacency matrix from a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph with edge attribute `"weight"`.

    Returns
    -------
    edges : np.ndarray, shape (m, 2), int64
        Sorted list of edges (u, v) with u < v (lexicographic order).
    adj_mat : np.ndarray, shape (n, n), float64
        Dense adjacency/weight matrix in node order `sorted(graph.nodes())`.

    Notes
    -----
    • The adjacency matrix uses the same node ordering used for the edge list
      extraction; downstream solvers should use this same ordering.
    """
    node_list = sorted(graph.nodes())
    adj_mat = nx.to_numpy_array(graph, node_list, weight="weight")

    graph_edges = [tuple(sorted(e)) for e in graph.edges()]
    graph_edges.sort(key=lambda x: (x[0], x[1]))

    return np.array(graph_edges, dtype=np.int64), adj_mat


# -----------------------------------------------------------------------------
# Colors -> paint swaps
# -----------------------------------------------------------------------------

def get_swaps(colors, car_pos: dict[int, tuple[int, int]]) -> tuple[list[int], int]:
    """
    Convert a car color assignment into a paint sequence and count paint swaps.

    Parameters
    ----------
    colors : array-like
        Per-car colors. Accepts either:
          • binary {0,1}, or
          • Ising-style {+1,-1} where -1 is treated as 0.
    car_pos : dict[int, (pos1, pos2)]
        Mapping car -> the two positions where it appears in the 2n-long sequence.

    Returns
    -------
    paint_sequence : list[int], length 2n
        The implied color along the production line positions.
        For each car i with color c, the two appearances are assigned:
            paint_sequence[pos1] = c
            paint_sequence[pos2] = 1 - c
    color_swaps : int
        Number of adjacent color changes along the paint_sequence, i.e.
        sum_{t=0}^{2n-2} |paint_sequence[t] - paint_sequence[t+1]|.
    """
    # Normalize Ising {-1,+1} -> {0,1}
    colors = [0 if c == -1 else int(c) for c in colors]

    n = len(colors)
    paint_sequence = [0 for _ in range(2 * n)]

    # Fill both occurrences for each car.
    for car, color in enumerate(colors):
        pos1, pos2 = car_pos[car]
        paint_sequence[pos1] = color
        paint_sequence[pos2] = 1 - color

    # Count adjacent swaps along the line.
    color_swaps = 0
    for i in range(len(paint_sequence) - 1):
        color_swaps += abs(paint_sequence[i] - paint_sequence[i + 1])

    return paint_sequence, color_swaps
