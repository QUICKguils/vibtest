""""Build and manipulate mechanical structures."""

from enum import Enum, auto
from typing import NamedTuple


class Point(NamedTuple):
    """A simple point in 3D cartesian space."""
    x: float
    y: float
    z: float


class Direction(Enum):
    """One of the three cartesian directions in space."""
    X = auto()
    Y = auto()
    Z = auto()


class Dof(NamedTuple):
    label: int
    pos: Point
    dir: Direction


class Structure:
    def __init__(self):
        self.dof_counter = 0

        self.dof_list: list[Dof] = []
        self.vertex_list: list[Point] = []
        self.edge_list: list[tuple[Point, Point]] = []

    def add_dof(self, pos: Point, dir: Direction):
        self.dof_counter += 1
        self.dof_list.append(Dof(label=self.dof_counter, pos=pos, dir=dir))

    def add_vertex(self, vertex: Point):
        self.vertex_list.append(vertex)

    def add_edge(self, vertex_tuple: tuple[Point, Point]):
        self.edge_list.append(vertex_tuple)

    def plot_geometry(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        for vertex in self.vertex_list:
            ax.scatter(vertex.x, vertex.y, vertex.z)
        for edge in self.edge_list:
            ax.plot([edge[0].x, edge[-1].x], [edge[0].y, edge[-1].y], [edge[0].z, edge[-1].z])

        fig.show()
