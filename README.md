# vibtest

Experimental modal analysis of a simplified aircraft structure.

This repository holds all the code written for the project carried out as part of the Vibration
Testing and Experimental Modal Analysis course (MECA0062-1), academic year 2024-2025.

## Disclaimer

The GitHub version of this repository does not hold the (fairly heavy) data recorded during the lab
sessions. Consequently, the code downloaded and installed from GitHub won't produce any results.

## Installation

From the top level directory (where this README lies),
create a virtual environment, and activate it.
```sh
# create a virtual environment
python -m venv .venv

# activate the virtual environment
source .venv/bin/activate  # on Unix/macOS
.venv\Scripts\activate      # on Windows
```

Still from the top level, install the package (optionally in editable mode, with the `-e` flag).
```sh
python -m pip install -e .
```

## Usage

The file `MECA0062_Ernotte.py` is meant to be directly executed with the python interpreter.
This will trigger the entire project computations.
```sh
python src/vibtest/MECA0062_Ernotte.py
```

In any cases, the code can be used like any other python package.
```python
import vibtest
# ...do whathever you want with the package

# Example 1
# Run the second part of the project.
from vibtest.project import part_2
sol = part_2.main()

# Example 2
# Check the coherences of the first lab session data.
from vibtest.project import statement
statement._inspect_coherences_lab_1()
```

## Project layout

The source code lies in `src/vibtest/`.
It contains the following packages and modules.
- Packages:
  - `project/` Code developed as part of the project.
- Modules:
  - `mplrc.py` Set some global Matplotlib parameters.
  - `structural.py` Build and manipulate mechanical structures.
  - `sdof.py` Single-degree-of-freedom identification techniques.
  - `mdof.py` Multiple-degree-of-freedom identification techniques.
