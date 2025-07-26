""" "Build and manipulate mechanical structures."""

from enum import Enum, auto
from typing import List, Tuple, NamedTuple


class Point(NamedTuple):
    """A simple point in 3D cartesian space."""

    x: float
    y: float
    z: float


class Direction(Enum):
    """One of the three cartesian directions in space."""

    x = auto()
    y = auto()
    z = auto()


class Dof(NamedTuple):
    label: int
    pos: Point
    dir: Direction


class Structure:
    """Mechanical structure."""

    def __init__(self):
        self.vertex_counter = 0
        self.dof_counter = 0

        self.accelerometer_list: List[Point] = []
        self.dof_list: List[Dof] = []
        self.dof_links: List[Tuple[Dof, Dof]] = []
        self.vertex_list: List[Point] = []
        self.vertex_links: List[Tuple[Point, Point]] = []

    def add_accelerometer(self, pos: Point):
        self.accelerometer_list.append(pos)

    def add_dof(self, pos: Point, dir: Direction):
        self.dof_counter += 1
        self.dof_list.append(Dof(label=self.dof_counter, pos=pos, dir=dir))

    def add_dof_link(self, dof_tuple: Tuple[Dof, Dof]):
        self.dof_links.append(dof_tuple)

    def add_vertex(self, vertex: Point):
        self.vertex_counter += 1
        self.vertex_list.append(vertex)

    def add_vertex_link(self, vertex_tuple: Tuple[Point, Point]):
        self.vertex_links.append(vertex_tuple)

    def plot_geometry(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        for accelerometer in self.accelerometer_list:
            ax.scatter(*accelerometer, color="C0", s=25)
        for edge in self.vertex_links:
            ax.plot(
                [edge[0].x, edge[-1].x],
                [edge[0].y, edge[-1].y],
                [edge[0].z, edge[-1].z],
                color="C7",
            )
        for dof in self.dof_list:
            ax.scatter(*dof.pos, color="C1", s=5)
        for edge in self.dof_links:
            ax.plot(
                [edge[0].pos.x, edge[-1].pos.x],
                [edge[0].pos.y, edge[-1].pos.y],
                [edge[0].pos.z, edge[-1].pos.z],
                color="C1",
            )

        ax.axis("off")
        ax.set_aspect("equal")
        ax.view_init(elev=25, azim=-135, roll=0)

        fig.show()
