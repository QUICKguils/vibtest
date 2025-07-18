"""Data manipulation utilities.

This module is in charge of handling the data
provided by the statement, the FEA and the two lab sessions.
"""

# TODO: plane structure/geometric utils go in this module

from scipy import io
import matplotlib.pyplot as plt

from vibtest.project import _PROJECT_PATH

LAB_DIR = [
    _PROJECT_PATH / "res" / "lab_1",
    _PROJECT_PATH / "res" / "lab_2"
]

NX_FREQENCIES = [  # identified from the initial NX model
    17.84,   # mode 1
    39.31,   # mode 2
    87.25,   # mode 3
    89.69,   # mode 4
    96.61,   # mode 5
    102.33,  # mode 6
    128.01,  # mode 7
    137.59,  # mode 8
    145.40,  # mode 9
    154.49,  # mode 10
    172.06,  # mode 11
    176.14,  # mode 12
    176.79,  # mode 13
]

def extract_measure(id_lab, id_measure):
    file_name = str(LAB_DIR[id_lab-1] / f"DPsv{id_measure:05}.mat")
    return io.loadmat(file_name)


def _inspect_coherences_lab_1():
    fig, axs = plt.subplots(4, 2, figsize=(8, 6))

    for id, ax in enumerate(axs.flat):
        data = extract_measure(1, id+1)

        c_12 = data["C1_2"]
        c_13 = data["C1_3"]
        c_14 = data["C1_4"]
        ax.plot(c_12[:, 0], c_12[:, -1])
        ax.plot(c_13[:, 0], c_13[:, -1])
        ax.plot(c_14[:, 0], c_14[:, -1])
        ax.set_title(f"run {id+1}")
        # ax.set_xlabel("Frequency (Hz)")
        # ax.set_ylabel("Coherence")

    fig.show()
