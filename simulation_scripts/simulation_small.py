"""
==============================================================================
Benchmark script for the Binary Paint Shop Problem (BPSP)

This script evaluates classical and quantum-inspired heuristics on small BPSP
instances and records the number of paint swaps obtained by each method.

Algorithms benchmarked here:
  • XQAOA (multi-start local optimization)
  • RQAOA
  • Greedy
  • Red-First
  • Recursive Greedy
  • Recursive Star Greedy (RSG) — saved separately

IMPORTANT NOTE
--------------
RSG results are written to a *separate directory* because they were generated
in a different run. This avoids re-organizing existing data and keeps all
downstream plotting scripts unchanged.

Output structure:
  simulation_data/small/
    ├── bpsp#{n}_{instance}.txt        (XQAOA + classical + RQAOA)
    └── RSG/
        └── bpsp#{n}.txt               (Recursive Star Greedy only)

============================================================================== 
"""

# Make repository root visible (simulation_scripts/ → repo root)
import sys
sys.path.append("..")

import os
import numpy as np
import multiprocessing
from functools import partial
from scipy import optimize

from BPSP import *
from XQAOA import *
from RQAOA import *
from BPSP_Greedy import *
from BPSP_Recursive import *
from BPSP_Star import *

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
os.makedirs("simulation_data/small", exist_ok=True)
os.makedirs("simulation_data/small/RSG", exist_ok=True)

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
for i in range(5, 125, 5):
    for j in range(51, 201):

        # ------------------------------------------------------------
        # Load BPSP instance
        # ------------------------------------------------------------
        with open(f"bpsp_data/small/bpsp#{i}_{j}.txt", "r") as f:
            seq = list(map(int, f.read().split()))

        car_pos = get_pos(np.array(seq))

        # ------------------------------------------------------------
        # Build Ising graph
        # ------------------------------------------------------------
        bpsp_graph = map_ising(seq)
        edges, adj_mat = get_edges_adj_mat(bpsp_graph)
        num_params = adj_mat.shape[0] + len(edges)

        # ------------------------------------------------------------
        # XQAOA: multi-start local optimization
        # ------------------------------------------------------------
        xqaoa = partial(XQAOA, edges, adj_mat)

        rand_angles = []
        for k in range(1, 101):
            np.random.seed(42 * k)
            rand_angles.append(np.random.uniform(0, 2 * np.pi, num_params))

        def optim(angles):
            res = optimize.minimize(
                xqaoa,
                angles,
                method="L-BFGS-B",
                options={"disp": False, "maxfun": 500_000},
            )
            _, swaps = get_color_swaps(edges, adj_mat, res.x, car_pos)
            print(f"XQAOA: {swaps} swaps | n={i}, instance={j}")
            return swaps

        with multiprocessing.Pool(31) as pool:
            xqaoa_swaps = pool.map(optim, rand_angles)

        # ------------------------------------------------------------
        # RQAOA
        # ------------------------------------------------------------
        if i == 5:
            gm = GraphManager(bpsp_graph, verbose=False)
            _, assignment = RQAOA(gm, 2)
        elif i == 10:
            gm = GraphManager(bpsp_graph, verbose=False)
            _, assignment = RQAOA(gm, 6)
        else:
            gm = GraphManager(bpsp_graph, verbose=False)
            _, assignment = RQAOA(gm, i - 8)

        _, swaps = get_swaps(list(assignment.values()), car_pos)
        print(f"RQAOA: {swaps} swaps | n={i}, instance={j}")
        xqaoa_swaps.append(swaps)

        # ------------------------------------------------------------
        # Classical baselines
        # ------------------------------------------------------------
        _, swaps = greedy_solver(seq, car_pos)
        print(f"Greedy: {swaps} swaps | n={i}, instance={j}")
        xqaoa_swaps.append(swaps)

        _, swaps = red_first_solver(seq)
        print(f"Red-First: {swaps} swaps | n={i}, instance={j}")
        xqaoa_swaps.append(swaps)

        _, swaps = recursive_greedy(seq, car_pos)
        print(f"Recursive Greedy: {swaps} swaps | n={i}, instance={j}")
        xqaoa_swaps.append(swaps)

        # ------------------------------------------------------------
        # Save combined results (unchanged format)
        # ------------------------------------------------------------
        swap_string = " ".join(str(s) for s in xqaoa_swaps)

        with open(f"simulation_data/small/bpsp#{i}_{j}.txt", "x") as f:
            f.write(swap_string)

    # -----------------------------------------------------------------------
    # Recursive Star Greedy (RSG) — saved separately
    # -----------------------------------------------------------------------
    swap_string = ""
    for j in range(1, 201):

        with open(f"bpsp_data/small/bpsp#{i}_{j}.txt", "r") as f:
            seq = list(map(int, f.read().split()))

        car_pos = get_pos(np.array(seq))

        _, swaps = recursive_star_greedy(seq, car_pos)
        print(f"RSG: {swaps} swaps | n={i}, instance={j}")
        swap_string += f"{swaps} "

    with open(f"simulation_data/small/RSG/bpsp#{i}.txt", "x") as f:
        f.write(swap_string)
