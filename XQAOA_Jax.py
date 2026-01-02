"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          JAX implementation of XQAOA objective and L-BFGS minimization
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides:
  • A JAX-differentiable implementation of the XQAOA p=1 objective for weighted
    graphs, written to be compatible with `jax.grad` / `jax.jit`.
  • A lightweight Optax L-BFGS minimization routine implemented using
    `jax.lax.while_loop`, suitable for fully JIT-able optimization loops.

Implementation Notes
--------------------
• Parameterization:
  The angle vector is packed as:
      angles = [beta_0, ..., beta_{n-1}, gamma_0, ..., gamma_{m-1}]
  where betas are node-local mixer angles and gammas are edge-local cost angles
  aligned with the row ordering of `edges`.

• Differentiability:
  The objective is computed using pure JAX primitives and reductions
  (`jnp.where`, `jnp.prod`, `jax.lax.scan`) to preserve differentiability.

• Performance:
  The objective is `@jax.jit` compiled. For repeated evaluations, keep `edges`
  and `adj_mat` as JAX arrays on-device to avoid host↔device transfer overhead.

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

import jax
import jax.numpy as jnp
import optax


@jax.jit
def XQAOA_Jax(edges: jnp.ndarray, adj_mat: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate the XQAOA p=1 objective for a weighted graph (JAX-differentiable).

    This function implements a differentiable, JIT-compilable cost function for
    an "extended QAOA" style parameterization where:
      • each node i has its own mixer angle beta_i, and
      • each edge (u, v) has its own cost angle gamma_{uv}.

    The input `angles` is packed as:
        angles[:num_nodes]  -> betas (shape: [n])
        angles[num_nodes:]  -> gammas (shape: [m]), aligned with `edges`

    Internally, an "angle matrix" Θ is formed such that:
      • Θ[i, i] = beta_i
      • Θ[u, v] = Θ[v, u] = gamma_{(u,v)}  for each edge in `edges`

    The objective is computed as a sum of per-edge contributions using `jax.lax.scan`
    (preferred over Python loops for JIT compilation). Each per-edge contribution
    depends on neighborhood products over cos(...) factors, implemented via masked
    products using `jnp.where(..., cos(...), 1)`.

    Parameters
    ----------
    edges : jnp.ndarray, shape (m, 2), integer dtype
        Edge list. Each row is [u, v]. The ordering must match the ordering of
        the edge-angle block of `angles` (gammas).
    adj_mat : jnp.ndarray, shape (n, n), float dtype
        Symmetric weighted adjacency matrix. Non-edges should have weight 0.
        The code uses (adj_mat[x, y] != 0) to infer connectivity.
    angles : jnp.ndarray, shape (n + m,), float dtype
        Concatenated angle vector: [betas (n), gammas (m)].

    Returns
    -------
    total_cost : jnp.ndarray scalar
        Total XQAOA objective value for the given angles.

    Notes
    -----
    • This implementation assumes an undirected graph.
    • If you have an adjacency matrix with very small weights for "non-edges",
      ensure they are exactly 0.0; otherwise, masks like (adj_mat != 0) will
      treat them as edges.
    • `jnp.prod` over many factors can underflow for large degrees; if you run
      into numerical issues on very dense graphs, consider accumulating sums of
      log-cos terms (log-domain) or using higher precision.
    """
    num_nodes = adj_mat.shape[0]

    # -------------------------------
    # 1) Unpack angles into blocks
    # -------------------------------
    # Node-local mixer angles (betas): shape (n,)
    betas = angles[:num_nodes]
    # Edge-local cost angles (gammas): shape (m,)
    gammas = angles[num_nodes:]

    # ------------------------------------------
    # 2) Build dense "angle matrix" Θ (n x n)
    # ------------------------------------------
    # Diagonal holds betas; off-diagonals are filled on the given edges with gammas.
    # Using `.at[...].set(...)` keeps this purely functional (JAX-friendly).
    angle_mat = jnp.diag(betas)
    angle_mat = angle_mat.at[edges[:, 0], edges[:, 1]].set(gammas)
    angle_mat = angle_mat.at[edges[:, 1], edges[:, 0]].set(gammas)

    # Precompute index helpers used in masking
    nodes = jnp.arange(num_nodes)
    ones = jnp.ones(num_nodes, dtype=adj_mat.dtype)

    # ------------------------------------------------------------------
    # Per-edge accumulation: scan over edges to keep JIT + grad friendly
    # ------------------------------------------------------------------
    def edge_cost_fn(total_cost, uv):
        """
        Add the contribution from a single edge (u, v) into the running total.

        Implemented as a scan body so the whole objective is compiled and
        differentiable without Python-side looping.
        """
        # uv is a length-2 array; force integer indices
        u, v = uv
        u = jnp.int32(u)
        v = jnp.int32(v)

        # Masks for neighbor sets:
        #   e_mask: neighbors of v excluding u
        #   d_mask: neighbors of u excluding v
        #   F_mask: common neighbors (triangles through u-v)
        e_mask = (adj_mat[v, :] != 0) & (nodes != u)
        d_mask = (adj_mat[u, :] != 0) & (nodes != v)
        F_mask = e_mask & d_mask

        # Masks for "non-triangle" incident edges (neighbors not in F)
        e_not_F_mask = e_mask & (~F_mask)
        d_not_F_mask = d_mask & (~F_mask)

        beta_u = angle_mat[u, u]
        beta_v = angle_mat[v, v]

        # ==============================================================
        # Term 1: "single-neighbor path" contribution
        # ==============================================================
        # Base factor: cos(2β_u) cos(2β_v) sin(w_uv * γ_uv)
        term1 = -jnp.cos(2 * beta_u) * jnp.cos(2 * beta_v) * jnp.sin(adj_mat[u, v] * angle_mat[u, v])

        # Neighbor product over v-side excluding u
        # We use a masked product: cos(...) where mask True, else multiply by 1
        term1_e = jnp.cos(2 * beta_u) * jnp.sin(2 * beta_v)
        term1_e = term1_e * jnp.prod(
            jnp.where(e_mask, jnp.cos(adj_mat[:, v] * angle_mat[:, v]), ones)
        )

        # Neighbor product over u-side excluding v
        term1_d = jnp.sin(2 * beta_u) * jnp.cos(2 * beta_v)
        term1_d = term1_d * jnp.prod(
            jnp.where(d_mask, jnp.cos(adj_mat[u, :] * angle_mat[u, :]), ones)
        )

        term1 = term1 * (term1_e + term1_d)

        # ==============================================================
        # Terms 2/3: triangle / non-triangle split structure
        # ==============================================================
        # Reuse the non-triangle products (they appear in both terms)
        e_not_F_term = jnp.prod(
            jnp.where(e_not_F_mask, jnp.cos(adj_mat[:, v] * angle_mat[:, v]), ones)
        )
        d_not_F_term = jnp.prod(
            jnp.where(d_not_F_mask, jnp.cos(adj_mat[u, :] * angle_mat[u, :]), ones)
        )

        # Triangle contributions: cos( (u,f)+(v,f) ) and cos( (u,f)-(v,f) )
        triangles_term1 = jnp.prod(
            jnp.where(
                F_mask,
                jnp.cos(adj_mat[u, :] * angle_mat[u, :] + adj_mat[v, :] * angle_mat[v, :]),
                ones,
            )
        )
        triangles_term2 = jnp.prod(
            jnp.where(
                F_mask,
                jnp.cos(adj_mat[u, :] * angle_mat[u, :] - adj_mat[v, :] * angle_mat[v, :]),
                ones,
            )
        )

        # -------------------------------
        # Term 2: symmetric triangle sum
        # -------------------------------
        term2 = 0.5 * jnp.sin(2 * beta_u) * jnp.sin(2 * beta_v)
        term2 = term2 * e_not_F_term * d_not_F_term * (triangles_term1 + triangles_term2)

        # ---------------------------------
        # Term 3: antisymmetric triangle diff
        # ---------------------------------
        term3 = -0.5 * jnp.cos(2 * beta_u) * jnp.cos(2 * beta_v) * jnp.sin(2 * beta_u) * jnp.sin(2 * beta_v)
        term3 = term3 * e_not_F_term * d_not_F_term * (triangles_term1 - triangles_term2)

        # Final weighted edge contribution
        cost_edge = adj_mat[u, v] * (term1 + term2 + term3)
        return total_cost + cost_edge, None

    total_cost, _ = jax.lax.scan(edge_cost_fn, 0.0, edges)
    return total_cost


def minimise(
    loss_fn,
    init_guess: jnp.ndarray,
    max_iter: int = 500,
    tolerance: float = 1e-3,
):
    """
    Minimize a scalar JAX loss function using Optax L-BFGS (JIT-friendly loop).

    This routine uses Optax's L-BFGS optimizer and runs the update loop using
    `jax.lax.while_loop` so that the entire optimization can be staged to XLA
    (i.e., compiled) if desired.

    Parameters
    ----------
    loss_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Scalar-valued function to minimize. Must be compatible with JAX tracing.
        Typically: `loss_fn = lambda angles: XQAOA_Jax(edges, adj_mat, angles)`
        (possibly with a sign flip depending on your objective convention).
    init_guess : jnp.ndarray
        Initial parameter vector.
    max_iter : int, default=500
        Maximum number of L-BFGS iterations.
    tolerance : float, default=1e-3
        Convergence threshold on the update L2 norm. The loop stops when
        ||updates|| < tolerance.

    Returns
    -------
    final_loss : jnp.ndarray scalar
        Final loss value.
    final_params : jnp.ndarray
        Optimized parameters.

    Notes
    -----
    • The stopping rule is based on the *update norm*, not gradient norm.
      If you prefer a gradient-based stopping criterion, replace update_norm
      with `jnp.linalg.norm(grad)` in the loop state.
    • `memory_size` controls the L-BFGS history length. Smaller memory reduces
      memory footprint and can be faster, but may converge more slowly.
    """
    # L-BFGS solver (limited-memory quasi-Newton)
    solver = optax.lbfgs(memory_size=10)
    opt_state = solver.init(init_guess)

    # Optax provides a state-aware (value, grad) helper for L-BFGS
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def cond_fn(state):
        step, params, opt_state, loss, update_norm = state
        return jnp.logical_and(step < max_iter, update_norm >= tolerance)

    def body_fn(state):
        step, params, opt_state, _, _ = state

        # Compute loss and gradient (and allow L-BFGS to use internal state)
        value, grad = value_and_grad(params, state=opt_state)

        # Update step; optax.lbfgs expects value_fn and current (value, grad)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=loss_fn
        )

        new_params = optax.apply_updates(params, updates)
        update_norm = jnp.linalg.norm(updates)

        return (step + 1, new_params, opt_state, value, update_norm)

    init_loss = loss_fn(init_guess)
    init_state = (0, init_guess, opt_state, init_loss, jnp.inf)

    final_step, final_params, final_opt_state, final_loss, final_update_norm = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_loss, final_params
