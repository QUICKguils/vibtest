"""Build and manipulate mechanical structures."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Point = npt.NDArray[np.float64]
Direction = npt.NDArray[np.int_]


@dataclass
class Dof:
    label: int
    pos: Point
    dir: Direction


class Structure:
    """Mechanical structure."""

    def __init__(self):
        self.vertex_counter = 0
        self.dof_counter = 0

        self.accelerometer_list: list[Point] = []
        self.vertex_list: list[Point] = []
        self.vertex_links: list[tuple[Point, Point]] = []
        self.dof_list: list[Dof] = []
        self.dof_links: list[tuple[Dof, Dof]] = []

    def add_accelerometer(self, pos: Point):
        self.accelerometer_list.append(pos)

    def add_vertex(self, pos: Point):
        self.vertex_counter += 1
        self.vertex_list.append(pos)

    def add_vertex_link(self, vertex_tuple: tuple[Point, Point]):
        self.vertex_links.append(vertex_tuple)

    def add_dof(self, pos: Point, dir: Direction):
        self.dof_counter += 1
        self.dof_list.append(Dof(label=self.dof_counter, pos=pos, dir=dir))

    def add_dof_link(self, dof_tuple: tuple[Dof, Dof]):
        self.dof_links.append(dof_tuple)

    def plot_geometry(self, plot_object=None):
        import matplotlib.pyplot as plt

        # Add capability to draw on an already existing plot
        if plot_object is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            fig, ax = plot_object

        for a in self.accelerometer_list:
            ax.scatter(*a, color="C0", s=25)
        for edge in self.vertex_links:
            ax.plot(
                [edge[0][0], edge[-1][0]],
                [edge[0][1], edge[-1][1]],
                [edge[0][2], edge[-1][2]],
                color="C7",
                # alpha=0.3,
            )
        for dof in self.dof_list:
            ax.scatter(*dof.pos, color="C1", s=5)
        for edge in self.dof_links:
            ax.plot(
                [edge[0].pos[0], edge[-1].pos[0]],
                [edge[0].pos[1], edge[-1].pos[1]],
                [edge[0].pos[2], edge[-1].pos[2]],
                color="C1",
            )

        ax.axis("off")
        ax.set_aspect("equal")
        ax.view_init(elev=25, azim=-135, roll=0)

        fig.show()
