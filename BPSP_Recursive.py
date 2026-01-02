"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          Recursive Greedy Heuristic for BPSP
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module implements the Recursive Greedy Heuristic for the Binary Paint Shop
Problem (BPSP). The algorithm operates by iteratively removing cars from the
sequence until a trivial instance remains, then reinserting the removed cars
in reverse order while assigning paint colors using local consistency rules.

The method is classical, deterministic, and runs in linear time. It serves as
a strong classical baseline against which quantum and quantum-inspired
heuristics (RQAOA, XQAOA) are benchmarked.

Algorithmic Idea
----------------
1) Represent the car sequence as a doubly-linked list with sentinel Head/Tail.
2) Repeatedly remove the *last car* (both occurrences) until only two cars remain.
3) Assign base colors to the remaining cars.
4) Reinsert removed cars in reverse order, assigning colors using local rules
   that minimize paint swaps and preserve feasibility.

License
-------
MIT License © 2026 V. Vijendran

==============================================================================
"""


# ---------------------------------------------------------------------------
# Linked-list primitives
# ---------------------------------------------------------------------------

class Car:
    """
    Node representing a single occurrence of a car in the sequence.

    Each car appears exactly twice in the BPSP sequence, and each occurrence
    is represented by a separate Car node in a doubly-linked list.
    """

    def __init__(self, car_num, occurrence, head):
        self.CAR_NUM = car_num          # Car label
        self.OCCURRENCE = occurrence    # 1 or 2 (first / second appearance)
        self.PAINT = None               # Assigned paint color (0 or 1)
        self.LEFT = None                # Pointer to left neighbor
        self.RIGHT = None               # Pointer to right neighbor
        self.HEAD = head                # Reference to list head (for size tracking)

    def detach(self):
        """
        Remove this node from the linked list.
        """
        self.LEFT.RIGHT = self.RIGHT
        self.RIGHT.LEFT = self.LEFT
        self.HEAD.SIZE -= 1

    def attach(self):
        """
        Reinsert this node between its LEFT and RIGHT neighbors.
        """
        self.RIGHT.LEFT = self.LEFT.RIGHT = self
        self.HEAD.SIZE += 1


class Head:
    """
    Sentinel node marking the start of the linked list.
    """

    def __init__(self, size):
        self.SIZE = size     # Number of nodes currently in the list
        self.RIGHT = None


class Tail:
    """
    Sentinel node marking the end of the linked list.
    """

    def __init__(self):
        self.LEFT = None


# ---------------------------------------------------------------------------
# Local coloring rules used during reinsertion
# ---------------------------------------------------------------------------

def rule_1(car_list, car_pos):
    """
    Rule 1.1:
    If both neighbors exist and have the same color, inherit that color.

    This avoids introducing a color swap locally.
    """
    left_car = car_list[car_pos].LEFT
    right_car = car_list[car_pos].RIGHT

    if isinstance(left_car, Car) and isinstance(right_car, Car):
        if left_car.PAINT == right_car.PAINT:
            return True
    return False


def rule_2(car_list, car_pos):
    """
    Rule 1.2:
    Handle boundary and partially-colored configurations.

    This rule triggers if:
      • The left neighbor is Head and the right neighbor is colored, or
      • The left neighbor is colored and the right neighbor is uncolored.
    """
    left_car = car_list[car_pos].LEFT
    right_car = car_list[car_pos].RIGHT

    if isinstance(left_car, Head) and isinstance(right_car, Car):
        return True
    if isinstance(left_car, Car) and isinstance(right_car, Car):
        if left_car.PAINT is not None and right_car.PAINT is None:
            return True
    return False


# ---------------------------------------------------------------------------
# Recursive Greedy Heuristic
# ---------------------------------------------------------------------------

def recursive_greedy(car_sequence, car_positions):
    """
    Recursive Greedy Heuristic for the Binary Paint Shop Problem.

    Overview
    --------
    The algorithm recursively simplifies the BPSP instance by removing cars
    until only a trivial core remains, assigns colors to this core, and then
    reconstructs the solution by reinserting cars using local greedy rules.

    Steps
    -----
    1) Construct a doubly-linked list of car occurrences with Head/Tail sentinels.
    2) While more than two cars remain:
         • Remove the last car (both occurrences).
         • Record the removal order.
    3) Assign base colors to the remaining two cars.
    4) Reinsert removed cars in reverse order, assigning colors using:
         • Rule 1.1 (neighbor agreement),
         • Rule 1.2 (boundary / partial information),
         • A fallback consistency rule.
    5) Read off the final paint sequence and count color swaps.

    Parameters
    ----------
    car_sequence : list[int]
        Length-2n sequence of car labels.
    car_positions : dict[int, tuple[int, int]]
        Mapping car -> (first_pos, second_pos) in `car_sequence`.

    Returns
    -------
    paint_sequence : list[int]
        Color assignment (0 or 1) for each position in the sequence.
    color_swaps : int
        Number of adjacent color changes.
    """

    # Initialize linked list with Head sentinel
    Car_List = [Head(len(car_sequence))]

    # Create Car nodes for each occurrence
    for pos, car in enumerate(car_sequence):
        pos1, pos2 = car_positions[car]
        occurrence = 1 if pos == pos1 else 2
        Car_List.append(Car(car, occurrence, Car_List[0]))

    # Append Tail sentinel
    Car_List.append(Tail())

    # Wire LEFT / RIGHT pointers
    for i in range(len(Car_List) - 1):
        Car_List[i].RIGHT = Car_List[i + 1]
    for i in range(len(Car_List) - 1, 0, -1):
        Car_List[i].LEFT = Car_List[i - 1]

    removed_cars = []

    # ---------------------------------------------------------
    # Recursive elimination phase
    # ---------------------------------------------------------
    while Car_List[0].SIZE > 2:
        last_car = Car_List[-1].LEFT
        last_car_num = last_car.CAR_NUM
        pos1, pos2 = car_positions[last_car_num]

        Car_List[pos1 + 1].detach()
        Car_List[pos2 + 1].detach()

        removed_cars.append(last_car_num)

    removed_cars.reverse()

    # ---------------------------------------------------------
    # Base assignment for trivial instance
    # ---------------------------------------------------------
    Car_List[0].RIGHT.PAINT = 0
    Car_List[-1].LEFT.PAINT = 1

    # ---------------------------------------------------------
    # Reconstruction phase
    # ---------------------------------------------------------
    for removed_car in removed_cars:
        pos1, pos2 = car_positions[removed_car]

        Car_List[pos1 + 1].attach()
        Car_List[pos2 + 1].attach()

        if rule_1(Car_List, pos1 + 1):
            Car_List[pos1 + 1].PAINT = Car_List[pos1 + 1].LEFT.PAINT
            Car_List[pos2 + 1].PAINT = 1 - Car_List[pos1 + 1].PAINT

        elif rule_2(Car_List, pos1 + 1):
            if isinstance(Car_List[pos1 + 1].LEFT, Head):
                Car_List[pos1 + 1].PAINT = Car_List[pos1 + 1].RIGHT.PAINT
            else:
                Car_List[pos1 + 1].PAINT = Car_List[pos1 + 1].LEFT.PAINT
            Car_List[pos2 + 1].PAINT = 1 - Car_List[pos1 + 1].PAINT

        else:
            # Fallback consistency rule
            if Car_List[pos1 + 1].LEFT.PAINT != Car_List[pos1 + 1].RIGHT.PAINT:
                Car_List[pos2 + 1].PAINT = Car_List[pos2 + 1].LEFT.PAINT
                Car_List[pos1 + 1].PAINT = 1 - Car_List[pos2 + 1].PAINT
            else:
                raise RuntimeError("Unexpected coloring configuration.")

    # Extract final paint sequence
    paint_sequence = [Car_List[i].PAINT for i in range(1, len(Car_List) - 1)]

    # Count swaps
    color_swaps = sum(
        abs(paint_sequence[i] - paint_sequence[i + 1])
        for i in range(len(paint_sequence) - 1)
    )

    return paint_sequence, color_swaps
