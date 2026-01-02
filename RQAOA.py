"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          Recursive QAOA utilities for weighted Max-Cut / Ising models
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides:
  • Fast closed-form p=1 QAOA expectation evaluators for weighted graphs
    (total cost-like objective and per-edge correlator-like scores).
  • Support routines for optimizing gamma via a reduced 1D objective and
    deriving the corresponding optimal beta analytically.
  • A single elimination step used in Recursive QAOA (RQAOA) and a driver
    that repeats eliminations before brute-forcing the residual instance.

Implementation Notes
--------------------
• Angle convention:
  This implementation uses trigonometric factors of the form cos(2*gamma*w)
  and sin(2*gamma*w). This is a valid convention as long as it matches your
  circuit parameterization / Hamiltonian scaling. Keep it consistent across
  the entire codebase and paper.

• Performance:
  The core inner-loop routines are Numba JIT-compiled with `@jit(nopython=True)`.

How to Cite
-----------
If you use this code in academic work, please cite the associated paper:
Classical and Quantum Heuristics for the Binary Paint Shop Problem - https://arxiv.org/abs/2509.15294

License
-------
MIT License © 2026 V. Vijendran

==============================================================================
"""

import sys
from functools import partial

import numpy as np
import networkx as nx
from numba import jit
from scipy import optimize
from numpy import sin, cos, pi

from utils import *  # expects GraphManager, extract_properties, etc.


# ---------------------------------------------------------------------------
# p=1 QAOA closed-form evaluators
# ---------------------------------------------------------------------------

@jit(nopython=True)
def QAOA_Expectation_Cost(edges, adj_mat, angles):
    """
    Compute a total p=1 QAOA objective for a weighted graph (implementation convention).

    This routine evaluates a closed-form p=1 expression by iterating over edges
    (u, v), computing a correlator-like quantity ZuZv (built from `term1 + term2`),
    and accumulating per-edge contributions:
        contribution(u,v) = 0.5 * w_uv * ZuZv

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list (u, v). Each edge should correspond to a nonzero weight in adj_mat.
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric weighted adjacency matrix (real weights).
    angles : array-like of length 2
        (gamma, beta) angles for p=1.

    Returns
    -------
    float
        Total objective value under this module's convention.

    Notes
    -----
    • Angle scaling:
      This implementation uses cos(2*gamma*w) and sin(2*gamma*w).
    • If you use a minimizer/maximizer downstream, ensure the sign convention
      matches what you intend to optimize.
    """
    gamma, beta = angles
    total_cost = 0.0

    for u, v in edges:
        # Neighbors: e = N(v)\{u}, d = N(u)\{v}
        eX = np.nonzero(adj_mat[v])[0]
        e = eX[np.where(eX != u)]
        dX = np.nonzero(adj_mat[u])[0]
        d = dX[np.where(dX != v)]

        # Common neighbors (triangles through edge u-v)
        F = np.intersect1d(e, d)

        # ----- term1 -----
        term1_cos1 = 1.0
        for x in e:
            term1_cos1 *= cos(2.0 * gamma * adj_mat[x, v])

        term1_cos2 = 1.0
        for y in d:
            term1_cos2 *= cos(2.0 * gamma * adj_mat[u, y])

        term1 = sin(4.0 * beta) * sin(2.0 * gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)

        # Non-triangle incident edges adjacent to u-v
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = list(set(e_edges_non_triangle + d_edges_non_triangle))

        # ----- term2 -----
        # Note: sign convention here matches your original file.
        term2 = -pow(sin(2.0 * beta), 2)
        for x, y in E:
            term2 *= cos(2.0 * gamma * adj_mat[x, y])

        triangle_1_terms = 1.0
        triangle_2_terms = 1.0
        for f in F:
            triangle_1_terms *= cos(2.0 * gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(2.0 * gamma * (adj_mat[u, f] - adj_mat[v, f]))

        term2 *= (triangle_1_terms - triangle_2_terms)

        ZuZv = term1 + term2
        total_cost += 0.5 * adj_mat[u, v] * ZuZv

    return total_cost


@jit(nopython=True)
def QAOA_Expectation_Edges(edges, adj_mat, angles):
    """
    Compute per-edge correlator-like scores at p=1.

    This is structurally identical to `QAOA_Expectation_Cost`, but instead of
    summing contributions across edges, it returns the computed ZuZv value
    for each edge (u, v). In RQAOA, these values are typically used to choose
    the edge with the largest |ZuZv| for (anti-)correlation/elimination.

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list (u, v).
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric weighted adjacency matrix.
    angles : array-like of length 2
        (gamma, beta) angles.

    Returns
    -------
    dict
        Mapping {(u, v): ZuZv} for edges provided.

    Notes
    -----
    ZuZv here is the internal correlator-like quantity used by this module.
    Ensure your reduction rule (correlate vs anti-correlate) is consistent
    with how ZuZv should be interpreted.
    """
    gamma, beta = angles
    edge_costs = {}

    for u, v in edges:
        eX = np.nonzero(adj_mat[v])[0]
        e = eX[np.where(eX != u)]
        dX = np.nonzero(adj_mat[u])[0]
        d = dX[np.where(dX != v)]

        F = np.intersect1d(e, d)

        term1_cos1 = 1.0
        for x in e:
            term1_cos1 *= cos(2.0 * gamma * adj_mat[x, v])
        term1_cos2 = 1.0
        for y in d:
            term1_cos2 *= cos(2.0 * gamma * adj_mat[u, y])
        term1 = sin(4.0 * beta) * sin(2.0 * gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)

        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = list(set(e_edges_non_triangle + d_edges_non_triangle))

        term2 = -pow(sin(2.0 * beta), 2)
        for x, y in E:
            term2 *= cos(2.0 * gamma * adj_mat[x, y])

        triangle_1_terms = 1.0
        triangle_2_terms = 1.0
        for f in F:
            triangle_1_terms *= cos(2.0 * gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(2.0 * gamma * (adj_mat[u, f] - adj_mat[v, f]))
        term2 *= (triangle_1_terms - triangle_2_terms)

        edge_costs[(u, v)] = term1 + term2

    return edge_costs


@jit(nopython=True)
def QAOA_Expectation_Coefficients(edges, adj_mat, gamma):
    """
    Compute coefficient terms (A, B) used to reduce the p=1 optimization.

    This routine computes two scalar coefficients (A, B) such that the p=1
    objective can be optimized over gamma first, and then beta is recovered
    analytically from (A, B).

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list (u, v).
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric weighted adjacency matrix.
    gamma : float
        Gamma parameter.

    Returns
    -------
    (float, float)
        (term_A, term_B) coefficients.
    """
    term_A = 0.0
    term_B = 0.0

    for u, v in edges:
        eX = np.nonzero(adj_mat[v])[0]
        e = eX[np.where(eX != u)]
        dX = np.nonzero(adj_mat[u])[0]
        d = dX[np.where(dX != v)]
        F = np.intersect1d(e, d)

        # A-term (sin part with neighbor cos products)
        term1_cos1 = 1.0
        for x in e:
            term1_cos1 *= cos(2.0 * gamma * adj_mat[x, v])
        term1_cos2 = 1.0
        for y in d:
            term1_cos2 *= cos(2.0 * gamma * adj_mat[u, y])

        term_A += 0.5 * adj_mat[u, v] * sin(2.0 * gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)

        # B-term (cos product with triangle split)
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = list(set(e_edges_non_triangle + d_edges_non_triangle))

        term2 = 0.5 * adj_mat[u, v]
        for x, y in E:
            term2 *= cos(2.0 * gamma * adj_mat[x, y])

        triangle_1_terms = 1.0
        triangle_2_terms = 1.0
        for f in F:
            triangle_1_terms *= cos(2.0 * gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(2.0 * gamma * (adj_mat[u, f] - adj_mat[v, f]))

        term_B += term2 * (triangle_1_terms - triangle_2_terms)

    return term_A, 0.5 * term_B


def gamma_cost(edges, adj_mat, gamma):
    """
    1D objective in gamma used to locate a good gamma efficiently.

    SciPy optimizers may pass `gamma` as a 1D ndarray; this helper normalizes
    it to a float, computes (A, B), and returns a scalar objective.

    Parameters
    ----------
    edges, adj_mat : see `QAOA_Expectation_Coefficients`
    gamma : float or numpy.ndarray
        Candidate gamma.

    Returns
    -------
    float
        Scalar objective to minimize.
    """
    if isinstance(gamma, np.ndarray):
        gamma = float(gamma[0])

    termA, termB = QAOA_Expectation_Coefficients(edges, adj_mat, gamma)
    R = np.sqrt(termA * termA + termB * termB)
    return -R - termB


def optimal_beta(edges, adj_mat, opt_gamma):
    """
    Analytic beta recovery from the coefficients (A, B) at an optimal gamma.

    Parameters
    ----------
    edges, adj_mat : see `QAOA_Expectation_Coefficients`
    opt_gamma : float
        Selected gamma.

    Returns
    -------
    float
        The corresponding beta (in radians).
    """
    termA, termB = QAOA_Expectation_Coefficients(edges, adj_mat, opt_gamma)
    alpha = np.arctan2(termA, termB)
    return (alpha + np.pi) / 4.0


@jit(nopython=True)
def get_max_frequency(edges, adj_mat):
    """
    Heuristic bound on dominant trig frequency of the p=1 objective.

    Used to set a reasonable grid-search resolution or initial guess scale.

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
    adj_mat : numpy.ndarray, shape (n, n)

    Returns
    -------
    float
        Maximum estimated frequency scale across edges.
    """
    max_freq = 0.0

    for u, v in edges:
        EX = np.nonzero(adj_mat[v])[0]
        e = EX[(EX != v) & (EX != u)]
        e_freq = 0.0
        for w in e:
            e_freq += np.abs(adj_mat[v, w])

        DX = np.nonzero(adj_mat[u])[0]
        d = DX[(DX != u) & (DX != v)]
        d_freq = 0.0
        for w in d:
            d_freq += np.abs(adj_mat[u, w])

        max_term1_freq = 2.0 * (adj_mat[u, v] + max(e_freq, d_freq))

        F = np.intersect1d(e, d)
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = list(set(e_edges_non_triangle + d_edges_non_triangle))

        leading_freq = 0.0
        for x, y in E:
            leading_freq += np.abs(adj_mat[x, y])

        tri1 = 0.0
        tri2 = 0.0
        for f in F:
            tri1 += np.abs(adj_mat[u, f] + adj_mat[v, f])
            tri2 += np.abs(adj_mat[u, f] - adj_mat[v, f])

        max_term2_freq = 2.0 * (leading_freq + max(tri1, tri2))
        edge_freq = max(max_term1_freq, max_term2_freq)

        if edge_freq > max_freq:
            max_freq = edge_freq

    return max_freq


# ---------------------------------------------------------------------------
# RQAOA elimination + driver
# ---------------------------------------------------------------------------

def eliminate_variable(graphmanager: GraphManager, n_samps):
    """
    Perform one RQAOA elimination step on the current reduced instance.

    Workflow
    --------
    1) Extract (edges, adj_mat) from the current reduced graph.
    2) Optimize gamma using either:
       • brute-force grid search if n_samps is provided, or
       • local continuous optimization otherwise (with a frequency-informed x0).
    3) Recover beta analytically from the (A, B) coefficients.
    4) Compute per-edge scores and pick the edge with maximum |score|.
    5) Correlate or anti-correlate based on the sign of that edge score.

    Parameters
    ----------
    graphmanager : GraphManager
        Manages the evolving reduced graph, logs, and elimination operations.
        Expected methods: correlate(edge), anti_correlate(edge), brute_force().
    n_samps : int or None
        Number of brute-force samples for gamma grid search. If None, use a
        local optimizer with an initial guess set by the estimated frequency.

    Returns
    -------
    None
        Mutates `graphmanager.reduced_graph` in-place.
    """
    red_edges, adj_mat = extract_properties(graphmanager)

    # Optimize gamma only (beta recovered analytically)
    qaoa_gamma_obj = partial(gamma_cost, red_edges, adj_mat)

    max_freq = get_max_frequency(red_edges, adj_mat)

    if n_samps is not None:
        # 1D brute grid search on gamma in [0, pi]
        opt_gamma = optimize.brute(qaoa_gamma_obj, ((0.0, pi),), Ns=n_samps, workers=1)
        opt_gamma = float(opt_gamma[0]) if isinstance(opt_gamma, np.ndarray) else float(opt_gamma)
    else:
        # Frequency-informed initial guess (guard against max_freq=0)
        initial_guess = 1.0 / (2.0 * max_freq + 1.0) if max_freq != 0 else 0.1
        res = optimize.minimize(
            qaoa_gamma_obj,
            x0=np.array([initial_guess], dtype=float),
            bounds=[(0.0, np.pi / 2.0)],
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-12},
        )
        opt_gamma = float(res.x[0])

    opt_beta = optimal_beta(red_edges, adj_mat, opt_gamma)
    graphmanager.optimal_angles[graphmanager.iter] = [opt_gamma, opt_beta]

    # Log cost at optimized angles (for diagnostics)
    qaoa_cost = QAOA_Expectation_Cost(red_edges, adj_mat, np.array([opt_gamma, opt_beta]))

    # Rank edges by maximum absolute correlator-like score
    edge_costs = QAOA_Expectation_Edges(red_edges, adj_mat, np.array([opt_gamma, opt_beta]))
    edge_costs = dict(sorted(edge_costs.items(), key=lambda kv: np.abs(kv[1]), reverse=True))

    edge, weight = next(iter(edge_costs.items()))
    sign = int(np.sign(sys.float_info.epsilon + weight))

    # Triangle diagnostics (often useful when interpreting failures)
    num_triangles_u = nx.triangles(graphmanager.reduced_graph)[edge[0]]
    num_triangles_v = nx.triangles(graphmanager.reduced_graph)[edge[1]]

    if sign < 0:
        msg1 = f"QAOA Cost = {qaoa_cost}. Anti-Correlating Edge {edge} with |score|={weight}."
        msg2 = f"Node {edge[0]} and {edge[1]} are in {num_triangles_u} and {num_triangles_v} triangles respectively."
        graphmanager.log[graphmanager.iter] += msg1 + "\n" + msg2 + "\n"
        if graphmanager.verbose:
            print(msg1)
            print(msg2)
        graphmanager.anti_correlate(edge)

    elif sign > 0:
        msg1 = f"QAOA Cost = {qaoa_cost}. Correlating Edge {edge} with |score|={weight}."
        msg2 = f"Node {edge[0]} and {edge[1]} are in {num_triangles_u} and {num_triangles_v} triangles respectively."
        graphmanager.log[graphmanager.iter] += msg1 + "\n" + msg2 + "\n"
        if graphmanager.verbose:
            print(msg1)
            print(msg2)
        graphmanager.correlate(edge)

    else:
        msg = f"Cannot correlate or anti-correlate edge {edge} for score {weight}."
        graphmanager.log[graphmanager.iter] += msg + "\n"
        if graphmanager.verbose:
            print(msg)


def RQAOA(graphmanager: GraphManager, n, n_samps=None):
    """
    Run RQAOA for up to `n` elimination steps, then brute-force the residual instance.

    Parameters
    ----------
    graphmanager : GraphManager
        Stateful manager for the reduced graph and reduction operations.
    n : int
        Maximum number of elimination steps to attempt. Stops early if no edges remain.
    n_samps : int or None
        If provided, use a brute-force grid of size `n_samps` for gamma search.
        Otherwise, use a local optimizer with a frequency-informed initial guess.

    Returns
    -------
    Any
        Whatever `graphmanager.brute_force()` returns (e.g., best assignment/value).

    Side Effects
    ------------
    • Mutates `graphmanager.reduced_graph` via correlate/anti_correlate.
    • Writes iteration logs and optimal angles into `graphmanager.log` and
      `graphmanager.optimal_angles`.
    """
    i = 0

    while i <= n:
        if graphmanager.reduced_graph.number_of_edges() == 0:
            break

        msg = (
            f"Iter {i}: Graph has {graphmanager.reduced_graph.number_of_nodes()} nodes "
            f"and {graphmanager.reduced_graph.number_of_edges()} edges remaining."
        )
        graphmanager.log[graphmanager.iter] = msg + "\n"
        if graphmanager.verbose:
            print(msg)

        eliminate_variable(graphmanager, n_samps)

        i += 1
        graphmanager.iter += 1

    graphmanager.log[graphmanager.iter] = "\nBrute-Forcing\n"
    if graphmanager.verbose:
        print("\nBrute-Forcing")

    return graphmanager.brute_force()
