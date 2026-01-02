"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          Numba-accelerated XQAOA objective and angle post-processing
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides:
  • A Numba JIT-compiled implementation of the XQAOA p=1 objective for weighted
    graphs using node-local angles (betas) and edge-local angles (gammas).
  • Angle “reduction” utilities that snap angles into canonical representatives
    and simplify edge angles (gammas) while compensating via discrete beta flips.
  • A gamma-propagation routine that attempts to remove unnecessary edge angles
    without worsening the objective by adjusting endpoint beta assignments.
  • Convenience wrappers to convert optimized angles into discrete car colors
    and evaluate paint swap counts through BPSP utilities.

Implementation Notes
--------------------
• Parameterization:
  angles = [beta_0, ..., beta_{n-1}, gamma_0, ..., gamma_{m-1}]
  where gamma_k corresponds to edges[k] = (u_k, v_k).

• Angle convention:
  This implementation uses trigonometric factors like cos(2*beta) and
  sin(adj_mat[u,v] * gamma_uv). This is a valid convention as long as it is
  consistent with your circuit/Hamiltonian scaling throughout the repo.

• Performance:
  The core objective and post-processing kernels are Numba JIT-compiled with
  `@jit(nopython=True)`. To keep Numba in nopython mode, inputs should be
  NumPy arrays with stable dtypes:
    - edges:    int64 array of shape (m, 2)
    - adj_mat:  float64 array of shape (n, n)
    - angles:   float64 array of length (n + m)

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
from numba import jit
from numpy import sin, cos

# NOTE:
#   This import is repo-specific. It is assumed that `BPSP.py` provides
#   utilities such as `get_swaps(colors, car_positions)`.
from BPSP import *  # noqa: F401,F403  (kept for backward compatibility / notebook usage)


# -----------------------------------------------------------------------------
# Core objective: XQAOA (Numba-accelerated)
# -----------------------------------------------------------------------------

@jit(nopython=True)
def XQAOA(edges: np.ndarray, adj_mat: np.ndarray, angles: np.ndarray) -> float:
    """
    Compute the XQAOA p=1 objective for a weighted graph.

    This implementation follows an “extended QAOA” parameterization:
      • each node i has its own mixer angle beta_i (stored on the diagonal),
      • each edge (u, v) has its own cost angle gamma_uv (stored off-diagonal).

    The input `angles` is packed as:
        angles[:n]   -> betas  (node-local angles)
        angles[n:]   -> gammas (edge-local angles, aligned with `edges` ordering)

    The function builds a dense angle matrix Θ where Θ[i,i]=beta_i and
    Θ[u,v]=Θ[v,u]=gamma_uv, then computes an edge-wise contribution based on:
      • products of cos(...) terms over neighbors of u and v,
      • “triangle” versus “non-triangle” decomposition via common neighbors.

    Parameters
    ----------
    edges : np.ndarray, shape (m, 2), int64
        Edge list (u, v). The i-th row corresponds to gamma_i in `angles[n+i]`.
    adj_mat : np.ndarray, shape (n, n), float64
        Symmetric weighted adjacency matrix. Non-edges should have weight 0.
    angles : np.ndarray, shape (n + m,), float64
        Concatenated parameters: [betas (n), gammas (m)].

    Returns
    -------
    float
        Total objective value (sum of per-edge contributions).

    Notes
    -----
    • This routine assumes edges match the nonzero structure in `adj_mat`.
    • Keep `edges`, `adj_mat`, and `angles` dtypes stable for best Numba performance.
    """
    num_nodes = adj_mat.shape[0]

    # -------------------------------
    # 1) Unpack betas/gammas
    # -------------------------------
    betas = angles[:num_nodes]
    gammas = angles[num_nodes:]

    # -------------------------------
    # 2) Build dense angle matrix Θ
    # -------------------------------
    angle_mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        angle_mat[i, i] = betas[i]
    for i, (u, v) in enumerate(edges):
        angle_mat[u, v] = gammas[i]
        angle_mat[v, u] = gammas[i]

    # Accumulate per-edge contributions
    edge_costs = {}

    # -----------------------------------------------------------------
    # For each edge (u,v), compute the structured terms term1/term2/term3
    # -----------------------------------------------------------------
    for u, v in edges:
        # Neighbors of v excluding u
        eX = np.nonzero(adj_mat[v])[0]
        e = eX[np.where(eX != u)]

        # Neighbors of u excluding v
        dX = np.nonzero(adj_mat[u])[0]
        d = dX[np.where(dX != v)]

        # Common neighbors (triangles through edge (u,v))
        F = np.intersect1d(e, d)

        # ==============================================================
        # Term 1: “single-neighbor path” contribution
        # ==============================================================
        term1 = -cos(2 * angle_mat[u, u]) * cos(2 * angle_mat[v, v]) * sin(adj_mat[u, v] * angle_mat[u, v])

        # v-side product
        term1_e = cos(2 * angle_mat[u, u]) * sin(2 * angle_mat[v, v])
        for w in e:
            term1_e *= cos(adj_mat[w, v] * angle_mat[w, v])

        # u-side product
        term1_d = sin(2 * angle_mat[u, u]) * cos(2 * angle_mat[v, v])
        for w in d:
            term1_d *= cos(adj_mat[u, w] * angle_mat[u, w])

        term1 *= (term1_e + term1_d)

        # ==============================================================
        # Term 2: triangle-symmetric contribution
        # ==============================================================
        e_not_F = [x for x in e if x not in F]
        d_not_F = [x for x in d if x not in F]

        term2 = 0.5 * sin(2 * angle_mat[u, u]) * sin(2 * angle_mat[v, v])

        # Products over non-triangle incident edges
        e_not_F_term = 1.0
        for w in e_not_F:
            e_not_F_term *= cos(adj_mat[w, v] * angle_mat[w, v])

        d_not_F_term = 1.0
        for w in d_not_F:
            d_not_F_term *= cos(adj_mat[u, w] * angle_mat[u, w])

        # Products over triangle neighbors (common neighbors)
        triangles_term1 = 1.0
        triangles_term2 = 1.0
        for f in F:
            triangles_term1 *= cos(adj_mat[u, f] * angle_mat[u, f] + adj_mat[v, f] * angle_mat[v, f])
            triangles_term2 *= cos(adj_mat[u, f] * angle_mat[u, f] - adj_mat[v, f] * angle_mat[v, f])

        term2 *= e_not_F_term * d_not_F_term * (triangles_term1 + triangles_term2)

        # ==============================================================
        # Term 3: triangle-antisymmetric contribution
        # ==============================================================
        term3 = -0.5 * cos(2 * angle_mat[u, u]) * cos(2 * angle_mat[v, v]) \
                * sin(2 * angle_mat[u, u]) * sin(2 * angle_mat[v, v])
        term3 *= e_not_F_term * d_not_F_term * (triangles_term1 - triangles_term2)

        # Final edge contribution
        edge_costs[(u, v)] = adj_mat[u, v] * (term1 + term2 + term3)

    # Sum all per-edge contributions
    total_cost = 0.0
    for val in edge_costs.values():
        total_cost += val

    return total_cost


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def approx_equal(num1: float, num2: float, epsilon: float = 1e-3) -> bool:
    """
    Check approximate equality within an absolute tolerance.

    Parameters
    ----------
    num1, num2 : float
        Values to compare.
    epsilon : float, default=1e-3
        Absolute tolerance.

    Returns
    -------
    bool
        True if |num1 - num2| <= epsilon.
    """
    return abs(num1 - num2) <= epsilon


# -----------------------------------------------------------------------------
# Angle post-processing: reduction + propagation
# -----------------------------------------------------------------------------

@jit(nopython=True)
def reduce_angles(edges: np.ndarray, adj_mat: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Canonicalize angles by snapping betas and simplifying gammas edge-by-edge.

    Goal
    ----
    The XQAOA objective often has symmetries / redundancies:
      • node angles beta_i can often be wrapped modulo π (depending on convention),
      • some edge angles gamma_uv can be “absorbed” by flipping endpoint betas
        among a discrete set (here: {π/4, 3π/4}) without worsening the cost.

    This routine:
      1) Wraps each beta into [0, π) and snaps values close to π/4 or 3π/4.
      2) For each gamma_uv:
           • if gamma is close to {0, π, 2π, 3π} (with edge-specific tolerances),
             snap it to that target;
           • otherwise, attempt to set gamma_uv = 0 and then brute-force over the
             four combinations of (beta_u, beta_v) ∈ {π/4, 3π/4}², selecting the
             best (lowest) objective value.

    Parameters
    ----------
    edges : np.ndarray, shape (m, 2), int64
        Edge list (u, v), aligned with the gamma block.
    adj_mat : np.ndarray, shape (n, n), float64
        Weighted adjacency matrix.
    angles : np.ndarray, shape (n + m,), float64
        Packed [betas, gammas] parameter vector.

    Returns
    -------
    np.ndarray
        A new (n + m,) angle vector with reduced/simplified parameters.

    Notes
    -----
    • This is a *greedy* per-edge simplification. It is intended as a
      post-processing step after an optimizer, to map solutions into a more
      interpretable/canonical form.
    • Because it calls XQAOA repeatedly, this routine can be expensive for large m.
      Use it selectively (e.g., only on the best solutions).
    """
    n = adj_mat.shape[0]
    m = edges.shape[0]

    betas = angles[:n].copy()
    gammas = angles[n:].copy()

    # 1) Wrap betas into [0, π) and snap near special points
    for i in range(n):
        b = betas[i]
        while b > np.pi:
            b -= np.pi
        while b < 0.0:
            b += np.pi

        if abs(b - (np.pi / 4)) <= 1e-3:
            betas[i] = np.pi / 4
        elif abs(b - (3 * np.pi / 4)) <= 1e-3:
            betas[i] = 3 * np.pi / 4
        else:
            betas[i] = b

    two_pi = 2.0 * np.pi

    # Targets for snapping gammas (mod 2π) and their tolerances
    targets = np.array((0.0, np.pi, 2 * np.pi, 3 * np.pi))
    epsilons = np.array((1e-1, 1e-2, 1e-2, 1e-2))

    # Workspace for packing (betas, gammas) into a single vector for evaluation
    new_angles = np.empty(n + m, dtype=angles.dtype)

    # Iterate edges and simplify each gamma
    for idx in range(m):
        u = edges[idx, 0]
        v = edges[idx, 1]

        # Wrap gamma into [0, 2π)
        gamma = gammas[idx] % two_pi

        # 2a) Snap to a nearby target if close enough
        snapped = False
        for j in range(targets.shape[0]):
            if abs(gamma - targets[j]) <= epsilons[j]:
                gamma = targets[j] % two_pi
                snapped = True
                break

        # 2b) If not snapped, attempt to eliminate this gamma (set to 0)
        if not snapped:
            # Pack current angles and evaluate baseline cost
            for i in range(n):
                new_angles[i] = betas[i]
            for i in range(m):
                new_angles[n + i] = gammas[i]
            old_cost = XQAOA(edges, adj_mat, new_angles)

            # Temporarily set this gamma to zero and search best beta flips
            gammas[idx] = 0.0
            best_u = betas[u]
            best_v = betas[v]
            best_cost = old_cost

            # Brute-force over the discrete beta choices
            for bu in (np.pi / 4, 3 * np.pi / 4):
                for bv in (np.pi / 4, 3 * np.pi / 4):
                    betas[u] = bu
                    betas[v] = bv

                    # Pack and evaluate
                    for i in range(n):
                        new_angles[i] = betas[i]
                    for i in range(m):
                        new_angles[n + i] = gammas[i]
                    new_cost = XQAOA(edges, adj_mat, new_angles)

                    # Accept improvements (or ties within a tiny tolerance)
                    if (new_cost < best_cost) or (abs(new_cost - best_cost) <= 1e-9):
                        best_cost = new_cost
                        best_u = bu
                        best_v = bv

            # Restore the best discrete beta choices for endpoints
            betas[u] = best_u
            betas[v] = best_v
            gamma = 0.0

        # Finalize this gamma
        gammas[idx] = gamma

    # Return concatenated output (fresh array)
    out = np.empty(n + m, dtype=angles.dtype)
    for i in range(n):
        out[i] = betas[i]
    for i in range(m):
        out[n + i] = gammas[i]
    return out


@jit(nopython=True)
def propagate_gammas(edges: np.ndarray, adj_mat: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Greedily attempt to remove gamma angles by compensating with discrete beta flips.

    For each edge gamma_uv, we test whether setting gamma_uv = 0 increases
    the objective value. If it does (i.e., worsens under your minimization
    convention), we try all four combinations of (beta_u, beta_v) ∈ {π/4, 3π/4}²
    to find a choice that restores or improves the objective.

    This routine is intended as a post-processing “cleanup” step after `reduce_angles`.

    Parameters
    ----------
    edges : np.ndarray, shape (m, 2)
        Edge list aligned with gammas.
    adj_mat : np.ndarray, shape (n, n)
        Weighted adjacency matrix.
    angles : np.ndarray, shape (n + m,)
        Packed [betas, gammas] vector.

    Returns
    -------
    np.ndarray
        Updated angle vector after attempting gamma removals.

    Notes
    -----
    • This is greedy and order-dependent.
    • It can be expensive: it calls XQAOA multiple times per edge.
    """
    n = adj_mat.shape[0]
    m = edges.shape[0]

    betas = angles[:n].copy()
    gammas = angles[n:].copy()

    new_angles = np.empty(n + m, dtype=angles.dtype)

    for idx in range(m):
        u = edges[idx, 0]
        v = edges[idx, 1]

        # Evaluate baseline cost
        for i in range(n):
            new_angles[i] = betas[i]
        for i in range(m):
            new_angles[n + i] = gammas[i]
        old_cost = XQAOA(edges, adj_mat, new_angles)

        # Try zeroing gamma_uv
        gammas[idx] = 0.0
        for i in range(n):
            new_angles[i] = betas[i]
        for i in range(m):
            new_angles[n + i] = gammas[i]
        cost0 = XQAOA(edges, adj_mat, new_angles)

        # If cost worsens, attempt to fix via discrete beta flips
        if cost0 > old_cost:
            best_u = betas[u]
            best_v = betas[v]
            best_cost = old_cost

            for bu in (np.pi / 4, 3 * np.pi / 4):
                for bv in (np.pi / 4, 3 * np.pi / 4):
                    betas[u] = bu
                    betas[v] = bv

                    for i in range(n):
                        new_angles[i] = betas[i]
                    for i in range(m):
                        new_angles[n + i] = gammas[i]
                    c = XQAOA(edges, adj_mat, new_angles)

                    if (c < best_cost) or (abs(c - best_cost) <= 1e-9):
                        best_cost = c
                        best_u = bu
                        best_v = bv

            betas[u] = best_u
            betas[v] = best_v

    out = np.empty(n + m, dtype=angles.dtype)
    for i in range(n):
        out[i] = betas[i]
    for i in range(m):
        out[n + i] = gammas[i]
    return out


# -----------------------------------------------------------------------------
# Application to BPSP: angles -> colors -> paint swaps
# -----------------------------------------------------------------------------

def get_color_swaps(edges, adj_mat, angles, car_positions):
    """
    Convert continuous XQAOA angles into a binary color assignment and compute paint swaps.

    Pipeline
    --------
    1) Canonicalize/simplify angles with `reduce_angles`.
    2) Further remove redundant gammas with `propagate_gammas`.
    3) Convert betas into binary colors via a thresholding rule:
           beta ≈ π/4  -> color 1
           otherwise   -> color 0
    4) Use BPSP helper `get_swaps(colors, car_positions)` to compute the paint sequence
       and the number of swaps.

    Parameters
    ----------
    edges, adj_mat, angles :
        XQAOA inputs as in `XQAOA`.
    car_positions :
        Problem-specific input passed through to `get_swaps`.

    Returns
    -------
    paint_sequence, color_swaps
        Outputs returned by `get_swaps`.
    """
    reduced_angles = reduce_angles(edges, adj_mat, angles)
    final_angles = propagate_gammas(edges, adj_mat, reduced_angles)

    betas = final_angles[:adj_mat.shape[0]]
    colors = [1 if approx_equal(beta, np.pi / 4) else 0 for beta in betas]

    paint_sequence, color_swaps = get_swaps(colors, car_positions)
    return paint_sequence, color_swaps


def get_color_swaps_and_icc(edges, adj_mat, angles, car_positions):
    """
    Same as `get_color_swaps`, but also return the inferred binary color vector.

    Returns
    -------
    colors, paint_sequence, color_swaps
        colors is a list[int] in {0,1}.
    """
    reduced_angles = reduce_angles(edges, adj_mat, angles)
    final_angles = propagate_gammas(edges, adj_mat, reduced_angles)

    betas = final_angles[:adj_mat.shape[0]]
    colors = [1 if approx_equal(beta, np.pi / 4) else 0 for beta in betas]

    paint_sequence, color_swaps = get_swaps(colors, car_positions)
    return colors, paint_sequence, color_swaps
