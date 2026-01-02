"""
==============================================================================

Title:             Classical and Quantum Heuristics for the Binary Paint Shop Problem
Subtitle:          Recursive Star-Greedy (RSG) heuristic for BPSP
Repository:        https://github.com/vijeycreative/XQAOA-BPSP
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module implements the Recursive Star-Greedy (RSG) heuristic for the Binary
Paint Shop Problem (BPSP). RSG is a refinement of Recursive Greedy: it performs
the same “peel-and-reinsert” recursion, but introduces an intermediate symbolic
state '*' (“star”) to defer decisions in ambiguous local configurations.

The core idea is:
  • Eliminate cars to reach a small base instance.
  • Reinsert cars in reverse order.
  • Use a hierarchy of local rules (A–F) to assign paints to the two occurrences.
  • Detect special “star patterns” and mark affected cars with '*' to postpone
    resolving them until later.
  • Finally resolve stars to concrete colors and compute swap count.

Star Semantics
--------------
A star '*' marks an occurrence whose final color is intentionally left undecided
at the time it is introduced. This is useful when local constraints allow multiple
choices with similar immediate cost, but different downstream impact.
A later pass resolves all stars consistently (here: a simple fixed resolution rule).

Implementation Notes
--------------------
• Data structure:
  The sequence is represented as a doubly-linked list using sentinel Head/Tail
  nodes so that removing and reinserting occurrences is O(1).

• Correctness:
  This heuristic always returns a feasible BPSP assignment (each car appears twice
  with opposite colors), assuming `car_positions` correctly maps each car label to
  its two occurrence indices in `car_sequence`.

• Performance:
  The algorithm runs in O(n) linked-list operations plus small constant-time rule
  checks per reinsertion (rules are local).

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


# ---------------------------------------------------------------------------
# Linked-list primitives
# ---------------------------------------------------------------------------

class Car:
    """
    Node representing a single occurrence of a car in the sequence.

    Each car appears exactly twice in the BPSP sequence; each occurrence is a
    separate node. During reconstruction, PAINT is assigned as:
      • 0 or 1 (concrete paint), or
      • '*' (star / deferred decision).
    """

    def __init__(self, car_num, occurrence, head):
        self.CAR_NUM = car_num          # Car label (0..n-1 typically)
        self.OCCURRENCE = occurrence    # 1 for first appearance, 2 for second
        self.PAINT = None               # None, 0, 1, or '*'
        self.LEFT = None
        self.RIGHT = None
        self.HEAD = head               # Reference to Head sentinel (size tracking)

    def detach(self):
        """Remove this node from the linked list in O(1)."""
        self.LEFT.RIGHT = self.RIGHT
        self.RIGHT.LEFT = self.LEFT
        self.HEAD.SIZE -= 1

    def attach(self):
        """
        Reinsert this node between its current LEFT and RIGHT pointers in O(1).

        Note: this assumes LEFT and RIGHT still refer to the original neighbors
        (which they do in this implementation because we never overwrite them
        during detach).
        """
        self.RIGHT.LEFT = self.LEFT.RIGHT = self
        self.HEAD.SIZE += 1


class Head:
    """Sentinel node marking the start of the linked list."""
    def __init__(self, size):
        self.SIZE = size     # Number of active (non-sentinel) nodes in the list
        self.RIGHT = None


class Tail:
    """Sentinel node marking the end of the linked list."""
    def __init__(self):
        self.LEFT = None


# ---------------------------------------------------------------------------
# Local configuration rules (A–F)
# These rules decide how to assign colors to the pair (c1, c2) when reinserted.
#
# Naming convention:
#   c1 = first occurrence Car node
#   c2 = second occurrence Car node
# In your reconstruction, c2 is attached first, then c1, matching your logic.
# ---------------------------------------------------------------------------

def rule_a(c1, c2):
    """
    Rule A (front pattern):
    Trigger when c1 is the first real node after Head and c2 follows immediately:
        Head -> c1 -> c2 -> ...
    """
    return isinstance(c1.LEFT, Head) and (c2.LEFT is c1)


def rule_b(c2):
    """
    Rule B (end pattern):
    Trigger when c2 is the last real node before Tail:
        ... -> c2 -> Tail
    """
    return isinstance(c2.RIGHT, Tail)


def rule_c(c2):
    """
    Rule C (sandwich same):
    Trigger when c2 sits between two real cars with the same concrete paint:
        L(color) -> c2 -> R(color), where color ∈ {0,1}.
    """
    L, R = c2.LEFT, c2.RIGHT
    return (
        isinstance(L, Car) and isinstance(R, Car)
        and L.PAINT in (0, 1) and R.PAINT in (0, 1)
        and L.PAINT == R.PAINT
    )


def rule_d(c2):
    """
    Rule D (sandwich different):
    Trigger when c2 sits between two real cars with different concrete paints:
        L(0/1) -> c2 -> R(1/0).
    """
    L, R = c2.LEFT, c2.RIGHT
    return (
        isinstance(L, Car) and isinstance(R, Car)
        and L.PAINT in (0, 1) and R.PAINT in (0, 1)
        and L.PAINT != R.PAINT
    )


def rule_e(c1: Car, c2: Car) -> bool:
    """
    Rule E (star-neighbour, mismatch case):

    Trigger when:
      • c2 has exactly one '*' neighbour and one real-coloured neighbour, and
      • that real neighbour’s color differs from c1’s only real neighbour’s color.

    Intuition:
      When c2 touches a star, the local context is “partially undecided”.
      If the non-star side disagrees with c1’s continuation color, we choose an
      assignment for (c1,c2) that aligns c1 with its right neighbor and sets c2 opposite.
    """
    L, R = c2.LEFT, c2.RIGHT

    # Identify which side is star; the other side must be real-coloured.
    if isinstance(L, Car) and L.PAINT == '*':
        other = R
    elif isinstance(R, Car) and R.PAINT == '*':
        other = L
    else:
        return False

    # In this construction, c1’s only “already-determined” neighbour is to its right.
    c1_neigh = c1.RIGHT

    return (
        isinstance(other, Car)
        and other.PAINT in (0, 1)
        and isinstance(c1_neigh, Car)
        and c1_neigh.PAINT in (0, 1)
        and other.PAINT != c1_neigh.PAINT
    )


def rule_f(c1: Car, c2: Car) -> bool:
    """
    Rule F (star-neighbour, match case):

    Same structure as Rule E, but the real-coloured neighbour of c2 matches
    c1’s only real neighbour.

    In this case, the algorithm additionally “forces” the starred car’s two
    occurrences to become concrete (0/1) immediately, to prevent inconsistent
    propagation of '*' across later insertions.
    """
    L, R = c2.LEFT, c2.RIGHT

    if isinstance(L, Car) and L.PAINT == '*':
        other = R
    elif isinstance(R, Car) and R.PAINT == '*':
        other = L
    else:
        return False

    c1_neigh = c1.RIGHT

    return (
        isinstance(other, Car)
        and other.PAINT in (0, 1)
        and isinstance(c1_neigh, Car)
        and c1_neigh.PAINT in (0, 1)
        and other.PAINT == c1_neigh.PAINT
    )


# ---------------------------------------------------------------------------
# “Star rules” detect special symmetric patterns and decide whether to mark
# a newly-inserted car with '*'. These are *meta-rules* applied after A–F.
# ---------------------------------------------------------------------------

def star_rule_same(c1: Car, c2: Car) -> bool:
    """
    Star-Rule 1 (same-neighbour-color symmetry):

    Returns True if BOTH occurrences (c1 and c2) are each surrounded by
    two real neighbours with the same paint, and moreover all four neighbour
    paints agree.

    This indicates a configuration where locally both occurrences are in a
    “uniform environment”, so committing early can be arbitrary; the heuristic
    defers by placing stars.
    """
    for car in (c1, c2):
        L, R = car.LEFT, car.RIGHT
        if not (isinstance(L, Car) and isinstance(R, Car)):
            return False
        if L.PAINT not in (0, 1) or R.PAINT not in (0, 1):
            return False
        if L.PAINT != R.PAINT:
            return False

    c1L, c1R = c1.LEFT, c1.RIGHT
    c2L, c2R = c2.LEFT, c2.RIGHT
    return (c1L.PAINT == c1R.PAINT == c2L.PAINT == c2R.PAINT)


def star_rule_diff(c1: Car, c2: Car) -> bool:
    """
    Star-Rule 2 (different-neighbour-color symmetry):

    Returns True if BOTH occurrences (c1 and c2) are each between two real
    neighbours of different colors.

    This indicates a “boundary-like” ambiguity where either choice produces a
    swap locally, so the algorithm may defer via stars.
    """
    for car in (c1, c2):
        L, R = car.LEFT, car.RIGHT
        if not (isinstance(L, Car) and isinstance(R, Car)):
            return False
        if L.PAINT not in (0, 1) or R.PAINT not in (0, 1):
            return False
        if L.PAINT == R.PAINT:
            return False
    return True


# ---------------------------------------------------------------------------
# Debug helpers (optional)
# ---------------------------------------------------------------------------

def get_list(head: Head):
    """
    Traverse from Head.RIGHT to Tail (exclusive) and return the list of Car nodes.
    """
    out = []
    node = head.RIGHT
    while not isinstance(node, Tail):
        out.append(node)
        node = node.RIGHT
    return out


def debug_print(head: Head) -> None:
    """
    Print the current linked-list sequence in human-readable form:
        CAR_NUM(PAINT) CAR_NUM(PAINT) ...

    Useful for debugging rule application and star propagation.
    """
    seq = get_list(head)
    if not seq:
        print("<empty>\n")
        return

    pieces = [f"{car.CAR_NUM}({car.PAINT})" for car in seq]
    print(" ".join(pieces))
    print("\n")


# ---------------------------------------------------------------------------
# Recursive Star-Greedy algorithm
# ---------------------------------------------------------------------------

def recursive_star_greedy(car_sequence, car_positions):
    """
    Recursive Star-Greedy (RSG) heuristic for BPSP.

    What it does
    ------------
    RSG follows the same recursion pattern as Recursive Greedy:

      1) Build a doubly-linked list of the 2n car occurrences.
      2) Repeatedly remove cars until a small base case remains.
      3) Solve the base case by direct assignment.
      4) Reinsert cars in reverse order.

    The key difference is the use of a third symbolic value '*':

      • During reinsertion, local rules A–F attempt to assign concrete colors
        (0/1) to the two occurrences of the car.
      • After inserting a car, the algorithm checks “star rules” that detect
        symmetry/ambiguity. When triggered, the algorithm marks BOTH occurrences
        of a certain car as '*' to defer that decision.
      • After all reinsertion, a final pass resolves all stars to concrete 0/1.

    Parameters
    ----------
    car_sequence : list[int]
        Length-2n car sequence.
    car_positions : dict[int, tuple[int, int]]
        car -> (pos1, pos2) indices into `car_sequence`.

    Returns
    -------
    paint_seq : list[int]
        Length-2n list of paint assignments (0/1) for each position.
    swaps : int
        Total number of adjacent paint changes.
    """

    # ---------------------------------------------------------
    # 1) Build the initial linked list (Head + Car nodes + Tail)
    # ---------------------------------------------------------
    Car_List = [Head(len(car_sequence))]

    for pos, c in enumerate(car_sequence):
        p1, p2 = car_positions[c]
        occ = 1 if pos == p1 else 2
        Car_List.append(Car(c, occ, Car_List[0]))

    Car_List.append(Tail())

    # Wire RIGHT pointers
    for i in range(len(Car_List) - 1):
        Car_List[i].RIGHT = Car_List[i + 1]
    # Wire LEFT pointers
    for i in range(len(Car_List) - 1, 0, -1):
        Car_List[i].LEFT = Car_List[i - 1]

    # ---------------------------------------------------------
    # 2) Peel off cars until only two occurrences remain
    #    (i.e., Head.SIZE counts remaining nodes in list)
    # ---------------------------------------------------------
    removed = []
    while Car_List[0].SIZE > 2:
        first = Car_List[0].RIGHT
        removed.append(first.CAR_NUM)

        p1, p2 = car_positions[first.CAR_NUM]
        Car_List[p1 + 1].detach()
        Car_List[p2 + 1].detach()

    removed.reverse()

    # ---------------------------------------------------------
    # 3) Base-case coloring: force two survivors to 0 and 1
    # ---------------------------------------------------------
    Car_List[0].RIGHT.PAINT = 0
    Car_List[-1].LEFT.PAINT = 1

    # ---------------------------------------------------------
    # 4) Reinsert removed cars and assign colors using A–F
    #    Then optionally introduce '*' using star rules.
    # ---------------------------------------------------------
    for cnum in removed:
        p1, p2 = car_positions[cnum]
        c1 = Car_List[p1 + 1]  # first occurrence node
        c2 = Car_List[p2 + 1]  # second occurrence node

        # Reinsert in the same order as your original code
        c2.attach()
        c1.attach()

        # debug_print(Car_List[0])

        # -------------------------
        # Apply local rule A–F
        # -------------------------
        if rule_a(c1, c2):
            # A: at the very front, mirror the next neighbor
            color = c2.RIGHT.PAINT
            c2.PAINT = color
            c1.PAINT = 1 - color

        elif rule_b(c2):
            # B: at the very end, force (c1,c2) = (0,1)
            c2.PAINT = 1
            c1.PAINT = 0

        elif rule_c(c2):
            # C: between same colors, match them to avoid a swap
            color = c2.LEFT.PAINT
            c2.PAINT = color
            c1.PAINT = 1 - color

        elif rule_d(c2):
            # D: between different colors, align c1 with its right neighbor
            color = c1.RIGHT.PAINT
            c1.PAINT = color
            c2.PAINT = 1 - color

        elif rule_e(c1, c2):
            # E: star-neighbour mismatch case
            color = c1.RIGHT.PAINT
            c1.PAINT = color
            c2.PAINT = 1 - color

        elif rule_f(c1, c2):
            # F: star-neighbour match case, then resolve the starred car immediately
            color = c1.RIGHT.PAINT
            c1.PAINT = color
            c2.PAINT = 1 - color

            # Identify which neighbor is the star-car
            star_car = c2.RIGHT if (isinstance(c2.RIGHT, Car) and c2.RIGHT.PAINT == '*') else c2.LEFT
            star_car_num = star_car.CAR_NUM

            # Resolve that car’s two occurrences consistently using occurrence index
            sp1, sp2 = car_positions[star_car_num]
            idx1, idx2 = sp1 + 1, sp2 + 1

            if star_car.OCCURRENCE == 1:
                Car_List[idx1].PAINT = 1 - color
                Car_List[idx2].PAINT = color
            else:
                Car_List[idx2].PAINT = 1 - color
                Car_List[idx1].PAINT = color

        else:
            raise RuntimeError(f"A–F rules failed for car={c1.CAR_NUM} occurrences.")

        # ---------------------------------------------------------
        # Star rule check (as in your original logic)
        #
        # You check a particular “candidate” early in the list:
        #   star_car = Head.RIGHT.RIGHT
        # and then test whether its two occurrences satisfy a star pattern.
        # If so, both occurrences get marked as '*'.
        # ---------------------------------------------------------
        star_car = Car_List[0].RIGHT.RIGHT
        sc_p1, sc_p2 = car_positions[star_car.CAR_NUM]
        B1, B2 = Car_List[sc_p1 + 1], Car_List[sc_p2 + 1]

        # Only consider star marking if the occurrences are not adjacent (your condition).
        if B1.RIGHT is not B2:
            if star_rule_same(B1, B2) or star_rule_diff(B1, B2):
                B1.PAINT = '*'
                B2.PAINT = '*'

    # ---------------------------------------------------------
    # 5) Resolve all stars
    #
    # Your current policy is simple: every starred car gets forced to (0,1)
    # for its (first, second) occurrences. This is deterministic and keeps
    # feasibility, though it may not be optimal for swaps (by design).
    # ---------------------------------------------------------
    for node in Car_List[1:-1]:
        if node.PAINT == '*':
            p1, p2 = car_positions[node.CAR_NUM]
            idx1, idx2 = p1 + 1, p2 + 1
            Car_List[idx1].PAINT = 0
            Car_List[idx2].PAINT = 1

    # ---------------------------------------------------------
    # 6) Extract paint sequence and count swaps
    # ---------------------------------------------------------
    paint_seq = [node.PAINT for node in Car_List[1:-1]]
    swaps = sum(abs(paint_seq[i] - paint_seq[i + 1]) for i in range(len(paint_seq) - 1))

    return paint_seq, swaps