"""Constant quantities and data manipulation utilities.

This module is in charge of handling the data
provided by the statement, the FEA and the two lab sessions.
"""

from scipy import io

from vibtest.project import _PROJECT_PATH
from vibtest.structural import Direction, Point, Structure

LAB_DIR = [
    _PROJECT_PATH / "res" / "lab_1",
    _PROJECT_PATH / "res" / "lab_2"
]

NX_FREQ = [  # identified from the initial NX model
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


def init_geometry(plane: Structure):
    # WARN: edge assignment not super duper robust
    # same culprit as for the bare_structure.m of my thvib project:
    # edge creation relies on the order of the implicitely created dof labels.

    # Fuselage
    plane.add_vertex(Point(   0,  50,   0))
    plane.add_vertex(Point(   0, -50,   0))
    plane.add_vertex(Point(   0, -50, 150))
    plane.add_vertex(Point(   0,  50, 150))

    plane.add_vertex(Point(1200,  50,   0))
    plane.add_vertex(Point(1200, -50,   0))
    plane.add_vertex(Point(1200, -50, 150))
    plane.add_vertex(Point(1200,  50, 150))

    plane.add_vertex_link((plane.vertex_list[0], plane.vertex_list[1]))
    plane.add_vertex_link((plane.vertex_list[1], plane.vertex_list[2]))
    plane.add_vertex_link((plane.vertex_list[2], plane.vertex_list[3]))
    plane.add_vertex_link((plane.vertex_list[3], plane.vertex_list[0]))

    plane.add_vertex_link((plane.vertex_list[4], plane.vertex_list[5]))
    plane.add_vertex_link((plane.vertex_list[5], plane.vertex_list[6]))
    plane.add_vertex_link((plane.vertex_list[6], plane.vertex_list[7]))
    plane.add_vertex_link((plane.vertex_list[7], plane.vertex_list[4]))

    plane.add_vertex_link((plane.vertex_list[0], plane.vertex_list[4]))
    plane.add_vertex_link((plane.vertex_list[1], plane.vertex_list[5]))
    plane.add_vertex_link((plane.vertex_list[2], plane.vertex_list[6]))
    plane.add_vertex_link((plane.vertex_list[3], plane.vertex_list[7]))

    # Wing
    plane.add_vertex(Point(750,  750, 0))
    plane.add_vertex(Point(750, -750, 0))
    plane.add_vertex(Point(850, -750, 0))
    plane.add_vertex(Point(850,  750, 0))

    plane.add_vertex_link((plane.vertex_list[8],  plane.vertex_list[9]))
    plane.add_vertex_link((plane.vertex_list[9],  plane.vertex_list[10]))
    plane.add_vertex_link((plane.vertex_list[10], plane.vertex_list[11]))
    plane.add_vertex_link((plane.vertex_list[11], plane.vertex_list[8]))

    # Horizontal tail
    plane.add_vertex(Point(  0,  250, 75))
    plane.add_vertex(Point(  0, -250, 75))
    plane.add_vertex(Point(100, -250, 75))
    plane.add_vertex(Point(100,  250, 75))

    plane.add_vertex_link((plane.vertex_list[12], plane.vertex_list[13]))
    plane.add_vertex_link((plane.vertex_list[13], plane.vertex_list[14]))
    plane.add_vertex_link((plane.vertex_list[14], plane.vertex_list[15]))
    plane.add_vertex_link((plane.vertex_list[15], plane.vertex_list[12]))

    # Vertical tail
    plane.add_vertex(Point(  0, 0, 150))
    plane.add_vertex(Point(100, 0, 150))
    plane.add_vertex(Point(100, 0, 350))
    plane.add_vertex(Point(  0, 0, 350))

    plane.add_vertex_link((plane.vertex_list[16], plane.vertex_list[17]))
    plane.add_vertex_link((plane.vertex_list[17], plane.vertex_list[18]))
    plane.add_vertex_link((plane.vertex_list[18], plane.vertex_list[19]))
    plane.add_vertex_link((plane.vertex_list[19], plane.vertex_list[16]))

    # Left nacelle
    plane.add_vertex(Point(775, 250,   0))
    plane.add_vertex(Point(775, 250, -55))
    plane.add_vertex(Point(825, 250, -55))
    plane.add_vertex(Point(825, 250,   0))

    plane.add_vertex_link((plane.vertex_list[20], plane.vertex_list[21]))
    plane.add_vertex_link((plane.vertex_list[21], plane.vertex_list[22]))
    plane.add_vertex_link((plane.vertex_list[22], plane.vertex_list[23]))
    plane.add_vertex_link((plane.vertex_list[23], plane.vertex_list[20]))

    # Right nacelle
    plane.add_vertex(Point(775, -250,   0))
    plane.add_vertex(Point(775, -250, -55))
    plane.add_vertex(Point(825, -250, -55))
    plane.add_vertex(Point(825, -250,   0))

    plane.add_vertex_link((plane.vertex_list[24], plane.vertex_list[25]))
    plane.add_vertex_link((plane.vertex_list[25], plane.vertex_list[26]))
    plane.add_vertex_link((plane.vertex_list[26], plane.vertex_list[27]))
    plane.add_vertex_link((plane.vertex_list[27], plane.vertex_list[24]))


def init_dofs(plane: Structure):
    # WARN: edge assignment not super duper robust
    # same culprit as for the bare_structure.m of my thvib project:
    # edge creation relies on the order of the implicitely created dof labels.

    # Left wing
    plane.add_dof(Point(850, 740, 0), Direction.z)
    plane.add_dof(Point(800, 740, 0), Direction.z)
    plane.add_dof(Point(750, 740, 0), Direction.z)
    plane.add_dof(Point(850, 640, 0), Direction.z)
    plane.add_dof(Point(800, 640, 0), Direction.z)
    plane.add_dof(Point(750, 640, 0), Direction.z)
    plane.add_dof(Point(850, 540, 0), Direction.z)
    plane.add_dof(Point(800, 540, 0), Direction.z)
    plane.add_dof(Point(750, 540, 0), Direction.z)
    plane.add_dof(Point(850, 440, 0), Direction.z)
    plane.add_dof(Point(800, 440, 0), Direction.z)
    plane.add_dof(Point(750, 440, 0), Direction.z)
    plane.add_dof(Point(850, 340, 0), Direction.z)
    plane.add_dof(Point(800, 340, 0), Direction.z)
    plane.add_dof(Point(750, 340, 0), Direction.z)
    plane.add_dof(Point(850, 240, 0), Direction.z)
    plane.add_dof(Point(800, 240, 0), Direction.z)
    plane.add_dof(Point(750, 240, 0), Direction.z)
    plane.add_dof(Point(850, 140, 0), Direction.z)
    plane.add_dof(Point(800, 140, 0), Direction.z)
    plane.add_dof(Point(750, 140, 0), Direction.z)

    plane.add_dof_link((plane.dof_list[0], plane.dof_list[1]))
    plane.add_dof_link((plane.dof_list[1], plane.dof_list[2]))
    plane.add_dof_link((plane.dof_list[0], plane.dof_list[3]))
    plane.add_dof_link((plane.dof_list[1], plane.dof_list[4]))
    plane.add_dof_link((plane.dof_list[2], plane.dof_list[5]))

    plane.add_dof_link((plane.dof_list[3], plane.dof_list[4]))
    plane.add_dof_link((plane.dof_list[4], plane.dof_list[5]))
    plane.add_dof_link((plane.dof_list[3], plane.dof_list[6]))
    plane.add_dof_link((plane.dof_list[4], plane.dof_list[7]))
    plane.add_dof_link((plane.dof_list[5], plane.dof_list[8]))

    plane.add_dof_link((plane.dof_list[6], plane.dof_list[7]))
    plane.add_dof_link((plane.dof_list[7], plane.dof_list[8]))
    plane.add_dof_link((plane.dof_list[6], plane.dof_list[9]))
    plane.add_dof_link((plane.dof_list[7], plane.dof_list[10]))
    plane.add_dof_link((plane.dof_list[8], plane.dof_list[11]))

    plane.add_dof_link((plane.dof_list[9],  plane.dof_list[10]))
    plane.add_dof_link((plane.dof_list[10], plane.dof_list[11]))
    plane.add_dof_link((plane.dof_list[9],  plane.dof_list[12]))
    plane.add_dof_link((plane.dof_list[10], plane.dof_list[13]))
    plane.add_dof_link((plane.dof_list[11], plane.dof_list[14]))

    plane.add_dof_link((plane.dof_list[12], plane.dof_list[13]))
    plane.add_dof_link((plane.dof_list[13], plane.dof_list[14]))
    plane.add_dof_link((plane.dof_list[12], plane.dof_list[15]))
    plane.add_dof_link((plane.dof_list[13], plane.dof_list[16]))
    plane.add_dof_link((plane.dof_list[14], plane.dof_list[17]))

    plane.add_dof_link((plane.dof_list[15], plane.dof_list[16]))
    plane.add_dof_link((plane.dof_list[16], plane.dof_list[17]))
    plane.add_dof_link((plane.dof_list[15], plane.dof_list[18]))
    plane.add_dof_link((plane.dof_list[16], plane.dof_list[19]))
    plane.add_dof_link((plane.dof_list[17], plane.dof_list[20]))

    plane.add_dof_link((plane.dof_list[18], plane.dof_list[19]))
    plane.add_dof_link((plane.dof_list[19], plane.dof_list[20]))

    # Right wing
    plane.add_dof(Point(850, -140, 0), Direction.z)
    plane.add_dof(Point(800, -140, 0), Direction.z)
    plane.add_dof(Point(750, -140, 0), Direction.z)
    plane.add_dof(Point(850, -240, 0), Direction.z)
    plane.add_dof(Point(800, -240, 0), Direction.z)
    plane.add_dof(Point(750, -240, 0), Direction.z)
    plane.add_dof(Point(850, -340, 0), Direction.z)
    plane.add_dof(Point(800, -340, 0), Direction.z)
    plane.add_dof(Point(750, -340, 0), Direction.z)
    plane.add_dof(Point(850, -440, 0), Direction.z)
    plane.add_dof(Point(800, -440, 0), Direction.z)
    plane.add_dof(Point(750, -440, 0), Direction.z)
    plane.add_dof(Point(850, -540, 0), Direction.z)
    plane.add_dof(Point(800, -540, 0), Direction.z)
    plane.add_dof(Point(750, -540, 0), Direction.z)
    plane.add_dof(Point(850, -640, 0), Direction.z)
    plane.add_dof(Point(800, -640, 0), Direction.z)
    plane.add_dof(Point(750, -640, 0), Direction.z)
    plane.add_dof(Point(850, -740, 0), Direction.z)
    plane.add_dof(Point(800, -740, 0), Direction.z)
    plane.add_dof(Point(750, -740, 0), Direction.z)

    plane.add_dof_link((plane.dof_list[21], plane.dof_list[22]))
    plane.add_dof_link((plane.dof_list[22], plane.dof_list[23]))
    plane.add_dof_link((plane.dof_list[21], plane.dof_list[24]))
    plane.add_dof_link((plane.dof_list[22], plane.dof_list[25]))
    plane.add_dof_link((plane.dof_list[23], plane.dof_list[26]))

    plane.add_dof_link((plane.dof_list[24], plane.dof_list[25]))
    plane.add_dof_link((plane.dof_list[25], plane.dof_list[26]))
    plane.add_dof_link((plane.dof_list[24], plane.dof_list[27]))
    plane.add_dof_link((plane.dof_list[25], plane.dof_list[28]))
    plane.add_dof_link((plane.dof_list[26], plane.dof_list[29]))

    plane.add_dof_link((plane.dof_list[27], plane.dof_list[28]))
    plane.add_dof_link((plane.dof_list[28], plane.dof_list[29]))
    plane.add_dof_link((plane.dof_list[27], plane.dof_list[30]))
    plane.add_dof_link((plane.dof_list[28], plane.dof_list[31]))
    plane.add_dof_link((plane.dof_list[29], plane.dof_list[32]))

    plane.add_dof_link((plane.dof_list[30], plane.dof_list[31]))
    plane.add_dof_link((plane.dof_list[31], plane.dof_list[32]))
    plane.add_dof_link((plane.dof_list[30], plane.dof_list[33]))
    plane.add_dof_link((plane.dof_list[31], plane.dof_list[34]))
    plane.add_dof_link((plane.dof_list[32], plane.dof_list[35]))

    plane.add_dof_link((plane.dof_list[33], plane.dof_list[34]))
    plane.add_dof_link((plane.dof_list[34], plane.dof_list[35]))
    plane.add_dof_link((plane.dof_list[33], plane.dof_list[36]))
    plane.add_dof_link((plane.dof_list[34], plane.dof_list[37]))
    plane.add_dof_link((plane.dof_list[35], plane.dof_list[38]))

    plane.add_dof_link((plane.dof_list[36], plane.dof_list[37]))
    plane.add_dof_link((plane.dof_list[37], plane.dof_list[38]))
    plane.add_dof_link((plane.dof_list[36], plane.dof_list[39]))
    plane.add_dof_link((plane.dof_list[37], plane.dof_list[40]))
    plane.add_dof_link((plane.dof_list[38], plane.dof_list[41]))

    plane.add_dof_link((plane.dof_list[39], plane.dof_list[40]))
    plane.add_dof_link((plane.dof_list[40], plane.dof_list[41]))

    # Left horizontal tail
    plane.add_dof(Point(100, 150, 75), Direction.z)
    plane.add_dof(Point( 50, 150, 75), Direction.z)
    plane.add_dof(Point(  0, 150, 75), Direction.z)
    plane.add_dof(Point(100, 100, 75), Direction.z)
    plane.add_dof(Point( 50, 100, 75), Direction.z)
    plane.add_dof(Point(  0, 100, 75), Direction.z)
    plane.add_dof(Point(100,  60, 75), Direction.z)
    plane.add_dof(Point( 50,  60, 75), Direction.z)
    plane.add_dof(Point(  0,  60, 75), Direction.z)

    plane.add_dof_link((plane.dof_list[42], plane.dof_list[43]))
    plane.add_dof_link((plane.dof_list[43], plane.dof_list[44]))
    plane.add_dof_link((plane.dof_list[42], plane.dof_list[45]))
    plane.add_dof_link((plane.dof_list[43], plane.dof_list[46]))
    plane.add_dof_link((plane.dof_list[44], plane.dof_list[47]))

    plane.add_dof_link((plane.dof_list[45], plane.dof_list[46]))
    plane.add_dof_link((plane.dof_list[46], plane.dof_list[47]))
    plane.add_dof_link((plane.dof_list[45], plane.dof_list[48]))
    plane.add_dof_link((plane.dof_list[46], plane.dof_list[49]))
    plane.add_dof_link((plane.dof_list[47], plane.dof_list[50]))

    plane.add_dof_link((plane.dof_list[48], plane.dof_list[49]))
    plane.add_dof_link((plane.dof_list[49], plane.dof_list[50]))

    # Right horizontal tail
    plane.add_dof(Point(100,  -60, 75), Direction.z)
    plane.add_dof(Point( 50,  -60, 75), Direction.z)
    plane.add_dof(Point(  0,  -60, 75), Direction.z)
    plane.add_dof(Point(100, -100, 75), Direction.z)
    plane.add_dof(Point( 50, -100, 75), Direction.z)
    plane.add_dof(Point(  0, -100, 75), Direction.z)
    plane.add_dof(Point(100, -150, 75), Direction.z)
    plane.add_dof(Point( 50, -150, 75), Direction.z)
    plane.add_dof(Point(  0, -150, 75), Direction.z)

    plane.add_dof_link((plane.dof_list[51], plane.dof_list[52]))
    plane.add_dof_link((plane.dof_list[52], plane.dof_list[53]))
    plane.add_dof_link((plane.dof_list[51], plane.dof_list[54]))
    plane.add_dof_link((plane.dof_list[52], plane.dof_list[55]))
    plane.add_dof_link((plane.dof_list[53], plane.dof_list[56]))

    plane.add_dof_link((plane.dof_list[54], plane.dof_list[55]))
    plane.add_dof_link((plane.dof_list[55], plane.dof_list[56]))
    plane.add_dof_link((plane.dof_list[54], plane.dof_list[57]))
    plane.add_dof_link((plane.dof_list[55], plane.dof_list[58]))
    plane.add_dof_link((plane.dof_list[56], plane.dof_list[59]))

    plane.add_dof_link((plane.dof_list[57], plane.dof_list[58]))
    plane.add_dof_link((plane.dof_list[58], plane.dof_list[59]))

    # Vertical tail
    plane.add_dof(Point(100, 0, 300), Direction.y)
    plane.add_dof(Point(50,  0, 300), Direction.y)
    plane.add_dof(Point(0,   0, 300), Direction.y)
    plane.add_dof(Point(100, 0, 250), Direction.y)
    plane.add_dof(Point(50,  0, 250), Direction.y)
    plane.add_dof(Point(0,   0, 250), Direction.y)
    plane.add_dof(Point(100, 0, 200), Direction.y)
    plane.add_dof(Point(50,  0, 200), Direction.y)
    plane.add_dof(Point(0,   0, 200), Direction.y)
    plane.add_dof(Point(100, 0, 160), Direction.y)
    plane.add_dof(Point(50,  0, 160), Direction.y)
    plane.add_dof(Point(0,   0, 160), Direction.y)

    plane.add_dof_link((plane.dof_list[60], plane.dof_list[61]))
    plane.add_dof_link((plane.dof_list[61], plane.dof_list[62]))
    plane.add_dof_link((plane.dof_list[60], plane.dof_list[63]))
    plane.add_dof_link((plane.dof_list[61], plane.dof_list[64]))
    plane.add_dof_link((plane.dof_list[62], plane.dof_list[65]))

    plane.add_dof_link((plane.dof_list[63], plane.dof_list[64]))
    plane.add_dof_link((plane.dof_list[64], plane.dof_list[65]))
    plane.add_dof_link((plane.dof_list[63], plane.dof_list[66]))
    plane.add_dof_link((plane.dof_list[64], plane.dof_list[67]))
    plane.add_dof_link((plane.dof_list[65], plane.dof_list[68]))

    plane.add_dof_link((plane.dof_list[66], plane.dof_list[67]))
    plane.add_dof_link((plane.dof_list[67], plane.dof_list[68]))
    plane.add_dof_link((plane.dof_list[66], plane.dof_list[69]))
    plane.add_dof_link((plane.dof_list[67], plane.dof_list[70]))
    plane.add_dof_link((plane.dof_list[68], plane.dof_list[71]))

    plane.add_dof_link((plane.dof_list[69], plane.dof_list[70]))
    plane.add_dof_link((plane.dof_list[70], plane.dof_list[71]))

    # Engines
    plane.add_dof(Point(800,  250, -55), Direction.y)
    plane.add_dof(Point(800, -250, -55), Direction.y)


def init_plane() -> Structure:
    plane = Structure()

    init_geometry(plane)
    init_dofs(plane)

    return plane

PLANE = init_plane()
N_DOF = len(PLANE.dof_list)
