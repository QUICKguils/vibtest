"""Trigger all the project code."""

from dataclasses import dataclass

import vibtest.project.preliminary_ema as pema
import vibtest.project.detailed_ema as dema
import vibtest.project.comparison as comp


@dataclass(frozen=True)
class Solution:
    pema: pema.Solution
    dema: dema.Solution
    comp: comp.Solution


def main():
    sol_pema = pema.main()
    sol_dema = dema.main()
    sol_comp = comp.main(sol_dema)

    return Solution(sol_pema, sol_dema, sol_comp)


if __name__ == '__main__':
    main()
