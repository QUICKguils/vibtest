# vibtest

Experimental modal analysis of a simplified aircraft structure.

This repository holds all the code written for the project carried out as part
of the Vibration Testing and Experimental Modal Analysis course (MECA0062-1),
academic year 2024-2025.

## Installation

From the top level directory (where this README lies),
create a virtual environment, and activate it.
```sh
# create a virtual environment
python -m venv venv

# activate the virtual environment
source .venv/bin/activate  # on Unix/macOS
venv\Scripts\activate      # on Windows
```

Still from the top level, install the package
(optionally in editable mode, with the `-e` flag).
```sh
python -m pip install -e .
```

## Usage

```python
import vibtest
# ...do whathever you want with the package
```

## Project layout

The source code lies in `src/vibtest/`.
It contains the following packages and modules.
