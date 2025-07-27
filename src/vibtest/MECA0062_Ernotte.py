"""Trigger all the project code."""

from vibtest import project


def main():
    project.preliminary_ema.main()
    project.detailed_ema.main()
    project.comparison.main()


if __name__ == '__main__':
    main()
