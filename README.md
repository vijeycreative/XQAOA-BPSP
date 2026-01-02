
# Code, Algorithms, and Numerical Data for the Binary Paint Shop Problem


This repository contains the source code, benchmark datasets, and numerical results used in the research paper:

> V Vijendran, Dax Enshan Koh, Ping Koy Lam, Syed M Assad  
> *Classical and Quantum Heuristics for the Binary Paint Shop Problem*  
> [https://arxiv.org/abs/2509.15294](https://arxiv.org/abs/2509.15294)


The repository provides all materials needed to reproduce and analyze the
numerical results presented in the paper, including:

1. **Source code** for all classical heuristics and quantum algorithms (Red-First, Greedy, Recursive Greedy, Recursive Star Greedy, RQAOA, and XQAOA)
2. **Benchmark datasets** containing pre-generated Binary Paint Shop Problem (BPSP) instances for small and large problem sizes
3. **Simulation results** stored in lightweight, machine-readable `.txt` format
4. **Benchmark scripts** used to generate the reported numerical results
5. **An example notebook** demonstrating end-to-end usage of all algorithms

This repository is intended to serve as a reference and benchmarking resource for researchers studying combinatorial optimization and variational quantum-inspired algorithms.

---

## Repository Overview
```
.  
├── bpsp_data/ # Pre-generated BPSP instances  
│  
├── simulation_data/ # Numerical benchmark results  
│ ├── small/ # Small-scale benchmark outputs (per-instance files)  
│ ├── XQAOA/ # XQAOA results (100 runs per instance)  
│ ├── greedy_bpsp#_.txt # Large-scale: Greedy results (one value per instance)  
│ ├── rf_bpsp#_.txt # Large-scale: Red-First results  
│ ├── rgreedy_bpsp#_.txt # Large-scale: Recursive Greedy results  
│ ├── rsgreedy_bpsp#_.txt # Large-scale: Recursive Star Greedy (RSG) results  
│ └── rqaoa_bpsp#*.txt # Large-scale: RQAOA results  
│  
├── simulation_scripts/ # Benchmark driver scripts + plotting notebooks  
│ ├── simulation_small.py  
│ ├── simulation_large.py  
│ ├── simulation_xqaoa.py  
│ ├── First_Plot.ipynb  
│ └── Second_Plot.ipynb  
│  
├── Example.ipynb # End-to-end example on a single instance  
│  
├── BPSP.py # BPSP → Ising mapping and utilities  
├── BPSP_Greedy.py # Greedy and Red-First heuristics  
├── BPSP_Recursive.py # Recursive Greedy heuristic  
├── BPSP_Star.py # Recursive Star Greedy (RSG)  
├── RQAOA.py # Recursive QAOA implementation  
├── XQAOA.py # NumPy/Numba-based XQAOA  
├── XQAOA_Jax.py # JAX-differentiable XQAOA (+ Optax optimizer)  
├── utils.py # Helper utilities (used primarily by RQAOA)  
└── README.md
```


## File and Folder Description

- **`BPSP.py`** - Core utilities for the Binary Paint Shop Problem, including: instance helpers, mapping BPSP sequences to Ising Hamiltonians, extracting `(edges, adj_mat)`, and post-processing utilities for computing paint swaps.

- **`BPSP_Greedy.py`** - Baseline classical heuristics:
  - Greedy solver
  - Red-First solver

- **`BPSP_Recursive.py`** - Recursive Greedy heuristic implementation.

- **`BPSP_Star.py`** - Recursive Star Greedy (RSG) heuristic implementation.

- **`RQAOA.py`** - Recursive QAOA (RQAOA): recursively eliminates variables using closed-form p = 1 QAOA correlators and solves a small residual instance exactly.

- **`utils.py`** - Helper functions (primarily supporting `RQAOA.py`), including graph utilities, elimination bookkeeping, and other shared routines used by the RQAOA pipeline.

- **`XQAOA.py`** and **`XQAOA_Jax.py`** - XQAOA implementations:
  - `XQAOA.py`: NumPy/Numba CPU implementation  
  - `XQAOA_Jax.py`: JAX-based differentiable implementation used for accelerated multi-start optimization

- **`simulation_scripts/`** - Benchmark driver scripts used to generate the numerical results, plus plotting notebooks.

- **`simulation_data/`** - All numerical benchmark outputs in plain-text format.

- **`Example.ipynb`** - Demonstrates how to:
  1. Load a BPSP instance  
  2. Solve it with classical heuristics  
  3. Solve it with RQAOA  
  4. Solve it with XQAOA  
---

## Notes on the Benchmark Data Structure

The directory structure inside `simulation_data/` may appear irregular. This is intentional.

- Different algorithms were benchmarked at different points in time.
- **XQAOA produces multiple solutions per instance** (typically **100 runs**), whereas the other algorithms produce **a single solution per instance**.
- **RSG was benchmarked separately** from other heuristics for small benchmarks.
- Reorganizing previously generated files would have required updating downstream plotting and analysis scripts.

As a result:
- XQAOA results are stored in a dedicated `simulation_data/XQAOA/` subdirectory.
- RSG results may appear in separate folders for some small-benchmark outputs.
- Large-scale classical baselines and RQAOA are stored as flat per-`n` text files (one integer per instance).

This structure preserves compatibility with the plotting notebooks and analysis pipeline used in the paper.

---

## Benchmark Scripts and Reproducibility

The benchmark scripts in `simulation_scripts/` are **not guaranteed to be bit-for-bit identical** to the exact scripts originally used to generate every file in `simulation_data/` (because some algorithms were run at different times).

However:
- They implement the same algorithms,
- follow the same evaluation protocol, and
- reproduce the same qualitative and quantitative trends.

They are therefore sufficient to reproduce the results and conclusions reported in the paper.

---

## Usage

### Example Notebook
Open the end-to-end usage example:
```bash
jupyter notebook Example.ipynb
```

### Running Small-Scale Benchmarks
```bash
python simulation_scripts/simulation_small.py
```
### Running Large-Scale Benchmarks (Classical + RQAOA)
```bash
python simulation_scripts/simulation_large.py
```
### Running Large-Scale XQAOA Benchmarks
```bash
python simulation_scripts/simulation_xqaoa.py <n> <instance_id>
```
---

## Citation

If you find this repository useful for your research or benchmarking, please cite the associated paper:

```bibtex
@article{vijendran2025classical,
  title={Classical and quantum heuristics for the binary paint shop problem},
  author={Vijendran, V and Koh, Dax Enshan and Lam, Ping Koy and Assad, Syed M},
  journal={arXiv preprint arXiv:2509.15294},
  year={2025}
}
```

---

## License

This project is released under the **MIT License**, which permits use, modification, and distribution with attribution.

```
MIT License © 2026 V. Vijendran
```

---
