"""
==============================================================================
Large-Scale Benchmark Script for the Binary Paint Shop Problem (BPSP)

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This script benchmarks *single-shot* algorithms on large BPSP instances and
stores results in the same "flat file per algorithm per N" format used by the
plotting pipeline.

Algorithms evaluated (one solution per instance):
  • RQAOA (recursive p=1 QAOA elimination + brute-force residual)
  • Red-First heuristic
  • Greedy heuristic
  • Recursive Greedy heuristic
  • Recursive Star Greedy heuristic (RSG)

Output Format (matches existing plotting code)
---------------------------------------------
For a fixed problem size n, each algorithm produces one file:

  simulation_data/
      rf_bpsp#{n}.txt
      greedy_bpsp#{n}.txt
      rgreedy_bpsp#{n}.txt
      rsgreedy_bpsp#{n}.txt
      rqaoa_bpsp#{n}.txt

Each file contains whitespace-separated integers:
  swap_count(instance_1) swap_count(instance_2) ... swap_count(instance_K)

Notes
-----
• XQAOA is intentionally excluded here because it produces *100 runs per instance*
  and is stored separately in an XQAOA subfolder.
• RQAOA elimination depth is chosen so the residual brute-force is feasible
  (typically leave ~8 variables).

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

from BPSP import *
from RQAOA import *
from BPSP_Greedy import *
from BPSP_Recursive import *
from BPSP_Star import *  


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
i = 4096                 # problem size (number of cars)
INSTANCE_START = 1       
INSTANCE_END = 50       

# Output directory (flat files, same as your screenshot)
OUT_DIR = "simulation_data"
os.makedirs(OUT_DIR, exist_ok=True)

# Output filepaths (one per algorithm per n)
out_paths = {
    "rqaoa":   os.path.join(OUT_DIR, f"rqaoa_bpsp#{i}.txt"),
    "greedy":  os.path.join(OUT_DIR, f"greedy_bpsp#{i}.txt"),
    "rf":      os.path.join(OUT_DIR, f"rf_bpsp#{i}.txt"),
    "rgreedy": os.path.join(OUT_DIR, f"rgreedy_bpsp#{i}.txt"),
    "rsg":     os.path.join(OUT_DIR, f"rsgreedy_bpsp#{i}.txt"),
}

# If you prefer strict "create new only", replace "w" with "x".
# Using "w" avoids crashing if you re-run this script.
for p in out_paths.values():
    if os.path.exists(p):
        print(f"[Warning] Overwriting existing file: {p}")


# ---------------------------------------------------------------------------
# Accumulators (one swap count per instance)
# ---------------------------------------------------------------------------
rqaoa_swaps   = []
greedy_swaps  = []
rf_swaps      = []
rgreedy_swaps = []
rsg_swaps     = []

# ---------------------------------------------------------------------------
# Main loop over instances
# ---------------------------------------------------------------------------
for j in range(INSTANCE_START, INSTANCE_END + 1):

    # -----------------------
    # Load BPSP sequence
    # -----------------------
    with open(f"bpsp_data/bpsp#{i}_{j}.txt", "r") as f:
        seq = list(map(int, f.read().split()))

    car_pos = get_pos(np.array(seq))

    # -----------------------
    # Greedy
    # -----------------------
    _, swaps = greedy_solver(seq, car_pos)
    greedy_swaps.append(swaps)
    print(f"Greedy found {swaps} swaps for n={i}, instance={j}.")

    # -----------------------
    # Red-First
    # -----------------------
    _, swaps = red_first_solver(seq)
    rf_swaps.append(swaps)
    print(f"Red-First found {swaps} swaps for n={i}, instance={j}.")

    # -----------------------
    # Recursive Greedy
    # -----------------------
    _, swaps = recursive_greedy(seq, car_pos)
    rgreedy_swaps.append(swaps)
    print(f"Recursive-Greedy found {swaps} swaps for n={i}, instance={j}.")

    # -----------------------
    # Recursive Star Greedy (RSG)
    # -----------------------
    _, swaps = recursive_star_greedy(seq, car_pos)
    rsg_swaps.append(swaps)
    print(f"RSG found {swaps} swaps for n={i}, instance={j}.")

    # -----------------------
    # RQAOA (map → Ising → solve)
    # -----------------------
    bpsp_graph = map_ising(seq)

    # Choose elimination count so residual is small enough for brute-force.
    # Your previous choice (i - 8) leaves 8 remaining variables.
    elim_steps = max(i - 8, 0)

    gm = GraphManager(bpsp_graph, verbose=False)
    best_cost, assignment = RQAOA(gm, elim_steps)

    # assignment is a dict {node: ±1} (or 0/1 depending on your GraphManager);
    # your existing pipeline uses get_swaps(list(assignment.values()), car_pos).
    _, swaps = get_swaps(list(assignment.values()), car_pos)
    rqaoa_swaps.append(swaps)
    print(f"RQAOA found {swaps} swaps for n={i}, instance={j}.")


# ---------------------------------------------------------------------------
# Save results (one file per algorithm, matching your plotting format)
# ---------------------------------------------------------------------------
with open(out_paths["rqaoa"], "w") as f:
    f.write(" ".join(map(str, rqaoa_swaps)))

with open(out_paths["greedy"], "w") as f:
    f.write(" ".join(map(str, greedy_swaps)))

with open(out_paths["rf"], "w") as f:
    f.write(" ".join(map(str, rf_swaps)))

with open(out_paths["rgreedy"], "w") as f:
    f.write(" ".join(map(str, rgreedy_swaps)))

with open(out_paths["rsg"], "w") as f:
    f.write(" ".join(map(str, rsg_swaps)))

print("\nSaved results:")
for k, p in out_paths.items():
    print(f"  {k:7s} → {p}")
