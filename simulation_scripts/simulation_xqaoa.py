"""
==============================================================================
XQAOA (JAX) Benchmark Script for the Binary Paint Shop Problem (BPSP)

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This script benchmarks the XQAOA algorithm implemented in JAX on individual
Binary Paint Shop Problem (BPSP) instances.

IMPORTANT DESIGN NOTE
---------------------
XQAOA is treated differently from all other algorithms in this repository:

  • For each problem instance (n, instance_id), XQAOA is run **100 times**
    from independent random initializations.
  • Each run produces one locally optimized solution.
  • The full distribution of 100 paint-swap counts is saved.

As a result:
  • XQAOA results are stored **separately** in an `XQAOA` subfolder.
  • This avoids mixing multi-solution outputs with single-solution algorithms
    (Greedy, RQAOA, Recursive Greedy, RSG, etc.).
  • Downstream plotting and analysis scripts can treat XQAOA as a distribution,
    rather than a single-point heuristic.

Output Format
-------------
For each instance, the file:

  simulation_data/XQAOA_Large/bpsp#{n}_{instance}.txt

contains 100 integers:
  [swap_1 swap_2 ... swap_100]

each corresponding to one independent XQAOA optimization run.

Performance Notes
-----------------
• JAX is used for full autodiff and JIT compilation.
• L-BFGS-style optimization is implemented via Optax.
• GPU/accelerator memory usage is limited via
  `XLA_PYTHON_CLIENT_MEM_FRACTION`.

License
-------
MIT License © 2026 V. Vijendran
==============================================================================
"""

# Make repository root visible (simulation_scripts/ → repo root)
import sys
sys.path.append("..")

import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from BPSP import *
from XQAOA import *
from XQAOA_Jax import *

# ---------------------------------------------------------------------------
# Limit JAX memory usage (important on shared GPUs)
# ---------------------------------------------------------------------------
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

# ---------------------------------------------------------------------------
# Expected usage:
#   python xqaoa_jax_benchmark.py <n> <instance_id>
#
# Example:
#   python xqaoa_jax_benchmark.py 2048 2
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------
    if len(sys.argv) < 3:
        print("Usage: python filename.py <n> <instance_id>")
        sys.exit(1)

    print("JAX devices available:")
    print(jax.devices())

    n = int(sys.argv[1])
    instance_id = int(sys.argv[2])

    # ------------------------------------------------------------
    # Load BPSP instance
    # ------------------------------------------------------------
    with open(f"bpsp_data/bpsp#{n}_{instance_id}.txt", "r") as f:
        seq = list(map(int, f.read().split()))

    car_pos = get_pos(np.array(seq))

    # ------------------------------------------------------------
    # Map BPSP → Ising / Max-Cut instance
    # ------------------------------------------------------------
    bpsp_graph = map_ising(seq)
    edges, adj_mat = get_edges_adj_mat(bpsp_graph)

    num_params = adj_mat.shape[0] + len(edges)

    # ------------------------------------------------------------
    # XQAOA (JAX) setup
    # ------------------------------------------------------------
    # Loss function: differentiable XQAOA objective
    loss_fn = partial(
        XQAOA_Jax,
        jnp.array(edges),
        jnp.array(adj_mat)
    )

    # JIT-compiled optimizer wrapper
    global_minimise = jax.jit(minimise, static_argnums=(0,))

    # ------------------------------------------------------------
    # Generate 100 random initializations
    # ------------------------------------------------------------
    rand_angles = []
    for k in range(1, 101):
        np.random.seed(42 * k)
        rand_angles.append(
            np.random.uniform(0, np.pi, num_params).astype(np.float32)
        )

    # ------------------------------------------------------------
    # Run XQAOA from each initialization
    # ------------------------------------------------------------
    xqaoa_solutions = []

    for angles in rand_angles:
        _, opt_angles = global_minimise(loss_fn, jnp.array(angles))

        # Decode optimized angles → paint sequence → swaps
        _, _, color_swaps = get_color_swaps_and_icc(
            edges,
            adj_mat,
            np.array(opt_angles),
            car_pos
        )

        print(
            f"XQAOA found a solution with {color_swaps} paint swaps "
            f"for n = {n}, instance {instance_id}."
        )

        xqaoa_solutions.append(color_swaps)

    # ------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------
    best_swaps = min(xqaoa_solutions)
    print(
        f"Best solution found by XQAOA has {best_swaps} "
        f"({best_swaps / n:.4f}) paint swaps "
        f"for n = {n}, instance {instance_id}."
    )

    # ------------------------------------------------------------
    # Save all 100 results (separate folder by design)
    # ------------------------------------------------------------
    os.makedirs("simulation_data/XQAOA", exist_ok=True)

    swap_string = " ".join(str(s) for s in xqaoa_solutions)

    with open(
        f"simulation_data/XQAOA/bpsp#{n}_{instance_id}.txt",
        "x"
    ) as f:
        f.write(swap_string)
