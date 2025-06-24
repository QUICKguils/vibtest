"""labdata -- Manipulate data from the two lab sessions."""

from scipy import io
import matplotlib.pyplot as plt

from vibtest import _ROOT_PATH

LAB_DIR = [
    _ROOT_PATH / "res" / "lab_1",
    _ROOT_PATH / "res" / "lab_2"
]


def extract_measure(id_lab, id_measure):
    file_name = str(LAB_DIR[id_lab-1] / f"DPsv{id_measure:05}")
    return io.loadmat(file_name)


def _inspect_coherences_lab_1():
    fig, axs = plt.subplots(4, 2, figsize=(8, 6), layout='constrained')

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
