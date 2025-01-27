# vibtest

Vibration testing of a plane structure.

This repository holds all the code written for the project carried out as part
of the Vibration Testing and Experimental Modal Analysis course (MECA0062-1),
academic year 2024-2025.

## Installation

Get the project source code. For example, you can clone this repo.
```sh
git clone --depth 1 https://github.com/QUICKguils/vibtest.git
```

From the top level directory (where this README lies), create a virtual
environment, and activate it.
```sh
# create a virtual environment
python -m venv venv

# activate the virtual environment
source .venv/bin/activate  # on Unix/macOS
venv\Scripts\activate      # on Windows
```

Still from the top level, install the package.
```sh
python -m pip install .
```

## Usage

```python
import vibtest
# ...do whathever you want with the package
```

## Project architecture

- `src/`:
  - `util/`: collection of utility functions that are used throughout the
    project.
- `res/`: contains the `.MAT` files that hold the experimental data measured
  during the two lab sessions.
- `out/`:
