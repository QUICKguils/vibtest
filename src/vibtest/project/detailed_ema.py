"""Detailed experimental modal analysis.

Executing the main function of this module will trigger the following tasks.
- Computation of the modal parameters of the plane under study.
- Construction of the corresponding modes shapes.
- Display of the computed results, if desired.
"""

from typing import NamedTuple


class Solution(NamedTuple):
    pass


def main(*, out_enabled=True):
    """Execute the detailed EMA."""
    if out_enabled:
        sol = Solution()
        print_solution(sol)


def print_solution(sol: Solution):
    print("=== Solutions for the detailed EMA ===")
