"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          Baseline constructive heuristics (Greedy, Red-First)
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module implements two simple baseline heuristics for constructing a feasible
two-color paint plan for a Binary Paint Shop Problem (BPSP) instance:

  • Greedy: assigns the first occurrence of each car the same color as its
    predecessor on the line (starting with color 0), and assigns the second
    occurrence the opposite color.
  • Red-First: assigns the first occurrence of each car color 1 and the second
    occurrence color 0.

Both routines return the full paint sequence (color per position along the line)
and the resulting number of paint swaps (adjacent color changes).

Notes
-----
• These baselines are intended for comparison against quantum-inspired and
  quantum algorithms in the accompanying paper/repository.

License
-------
MIT License © 2026 V. Vijendran

==============================================================================
"""

from __future__ import annotations


def greedy_solver(car_sequence, car_pos):
    """
    Greedy constructive heuristic for BPSP.

    Strategy
    --------
    Scan the sequence left-to-right.
      • The first time a car appears, paint it with the SAME color as the
        immediately previous position (to avoid introducing a swap), except
        the very first position which is set to color 0.
      • The second time the same car appears, paint it with the OPPOSITE color
        of its first occurrence (enforcing the BPSP constraint).

    Parameters
    ----------
    car_sequence : sequence[int]
        Length 2n sequence of car labels (typically 0..n-1), each appearing twice.
    car_pos : dict[int, tuple[int, int]]
        Mapping car -> (first_position, second_position) in `car_sequence`.
        This is used to retrieve the color used on the first occurrence when
        processing the second occurrence.

    Returns
    -------
    paint_sequence : list[int]
        Length 2n list with entries in {0,1} giving the color at each position.
    color_swaps : int
        Number of adjacent color changes along `paint_sequence`, i.e.
        sum_{t=0}^{2n-2} |paint_sequence[t] - paint_sequence[t+1]|.
    """
    len_sequence = len(car_sequence)

    # Track whether we've already assigned a color to the first occurrence of each car.
    # NOTE: This assumes car labels are usable as dict keys and that every label appears.
    cars_painted = {car: False for car in set(car_sequence)}

    paint_sequence = []

    for car in car_sequence:
        if not cars_painted[car]:
            # First occurrence: match previous color if possible (minimizes immediate swaps).
            if not paint_sequence:
                paint_sequence.append(0)          # starting color convention
            else:
                paint_sequence.append(paint_sequence[-1])

            cars_painted[car] = True
        else:
            # Second occurrence: must be the opposite of its first occurrence.
            first_idx = car_pos[car][0]
            paint_sequence.append(1 - paint_sequence[first_idx])

    # Count adjacent swaps along the line.
    color_swaps = 0
    for i in range(len_sequence - 1):
        color_swaps += abs(paint_sequence[i] - paint_sequence[i + 1])

    return paint_sequence, color_swaps


def red_first_solver(car_sequence):
    """
    Red-First baseline heuristic for BPSP.

    Strategy
    --------
    Scan the sequence left-to-right.
      • First occurrence of each car -> color 1 ("red")
      • Second occurrence of each car -> color 0 ("blue")

    This is a very simple, fully deterministic baseline. In many instances it
    performs worse than the Greedy heuristic, but it is useful as a sanity-check
    baseline and for illustrating the value of better heuristics.

    Parameters
    ----------
    car_sequence : sequence[int]
        Length 2n sequence of car labels, each appearing twice.

    Returns
    -------
    paint_sequence : list[int]
        Length 2n list with entries in {0,1} giving the color at each position.
    color_swaps : int
        Number of adjacent color changes along `paint_sequence`.
    """
    len_sequence = len(car_sequence)

    # Track whether we've seen each car already.
    cars_painted = {car: False for car in set(car_sequence)}

    paint_sequence = []

    for car in car_sequence:
        if not cars_painted[car]:
            paint_sequence.append(1)   # first occurrence -> red
            cars_painted[car] = True
        else:
            paint_sequence.append(0)   # second occurrence -> blue

    # Count adjacent swaps.
    color_swaps = 0
    for i in range(len_sequence - 1):
        color_swaps += abs(paint_sequence[i] - paint_sequence[i + 1])

    return paint_sequence, color_swaps
