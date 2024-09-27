import numpy as np
from numpy import cos, sin, pi


def T(alpha):
    T = np.array([
        [cos(alpha), -sin(alpha), 0],
        [sin(alpha), cos(alpha), 0],
        [0, 0, 1]
    ])
    return T


# RECORDAR QUE LOS EJES LOCALES POSITIVOS SON X= DERECHA, Y=ARRIBA, M=ANTIHORARIO
def K11(EA, EI, L, alpha):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    k11 = np.array([[EA / L, 0, 0],
                    [0, 12 * EI / pow(L, 3), 6 * EI / pow(L, 2)],
                    [0, 6 * EI / pow(L, 2), 4 * EI / L]
                    ])
    glob_coord = T(alpha)
    return glob_coord @ k11 @ np.transpose(glob_coord)


def K12(EA, EI, L, alpha):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    k12 = np.array([[-EA / L, 0, 0],
                    [0, -12 * EI / pow(L, 3), 6 * EI / pow(L, 2)],
                    [0, -6 * EI / pow(L, 2), 2 * EI / L]
                    ])
    glob_coord = T(alpha)
    return glob_coord @ k12 @ np.transpose(glob_coord)


def K21(EA, EI, L, alpha):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    k21 = np.array([[-EA / L, 0, 0],
                    [0, -12 * EI / pow(L, 3), -6 * EI / pow(L, 2)],
                    [0, 6 * EI / pow(L, 2), 2 * EI / L]
                    ])
    glob_coord = T(alpha)
    return glob_coord @ k21 @ np.transpose(glob_coord)


def K22(EA, EI, L, alpha):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    k22 = np.array([[EA / L, 0, 0],
                    [0, 12 * EI / pow(L, 3), -6 * EI / pow(L, 2)],
                    [0, -6 * EI / pow(L, 2), 4 * EI / L]
                    ])
    glob_coord = T(alpha)
    return glob_coord @ k22 @ np.transpose(glob_coord)


if __name__=='__main__':
    np.set_printoptions(threshold=np.inf, linewidth=200)
    # N, m
    # initial properties

    L_viga = 5
    L_pilar = 4
    alpha_pilar1 = pi / 2


    # element properties
    EA = 5E9
    EI = 1E8
    # element matrices
    # first column
    K11_31 = K11(EA, EI, L_pilar, alpha_pilar1)
    K12_31 = K12(EA, EI, L_pilar, alpha_pilar1)
    K21_31 = K21(EA, EI, L_pilar, alpha_pilar1)
    K22_31 = K22(EA, EI, L_pilar, alpha_pilar1)

    # beam
    K11_12 = K11(EA, EI, L_viga, 0)
    K12_12 = K12(EA, EI, L_viga, 0)
    K21_12 = K21(EA, EI, L_viga, 0)
    K22_12 = K22(EA, EI, L_viga, 0)

    # second column
    K11_42 = K11(EA, EI, L_pilar, alpha_pilar1)
    K12_42 = K12(EA, EI, L_pilar, alpha_pilar1)
    K21_42 = K21(EA, EI, L_pilar, alpha_pilar1)
    K22_42 = K22(EA, EI, L_pilar, alpha_pilar1)

    # zero matrix
    zero = np.zeros((3, 3))
    # stiffness matrix
    stiff_matrix = np.block([
        [K11_12 + K22_31, K12_12, K21_31, zero],
        [K21_12, K22_12 + K22_42, zero, K21_42],
        [K12_31, zero, K11_31, zero],
        [zero, K12_42, zero, K11_42]
    ])

    print("stiffness matrix:\n", stiff_matrix)

    force_vector = np.array([
        0, 0, 10E3,
        1, 0, 20E3,
        1, 1, 1,
        1, 1, 1
    ])

    reduced_clm_rows = [0, 1, 2, 4, 5]  # accordint to zeros in displacements vectors
    reduced_stiff_matrix = stiff_matrix[np.ix_(reduced_clm_rows, reduced_clm_rows)]
    print(" reduced stiffness matrix:\n", reduced_stiff_matrix)

    reduced_force_vector = force_vector[reduced_clm_rows]  # flatten the vstack
    print("reduced force vector:\n", reduced_force_vector)

    reduced_displ_vector = np.linalg.solve(reduced_stiff_matrix, reduced_force_vector)
    print("reduced displacements:\n", reduced_displ_vector)

    disp_list = []
    # current item in reduced vector index
    red_vect_index = 0
    for i in range(0, len(force_vector)):
        # if index is in extracted rows and columns
        if i in reduced_clm_rows:
            disp_list.append(reduced_displ_vector[red_vect_index])
            red_vect_index += 1
        else:
            disp_list.append(0)

    displ_vector = np.array(disp_list)
    print("displacemten vector:\n", displ_vector)

    result_force_vector = stiff_matrix @ displ_vector
    print("result force vector:\n", result_force_vector)

# inner forces in each element
    # T transpose for first column
    print(T(alpha_pilar1))
    TT1 = np.transpose(T(alpha_pilar1))
    # T transpose for beam
    TT2 = np.transpose(T(0))

    # internal forces in each barr element
    p3_31 = np.block([TT1 @ K11_31, TT1 @ K12_31]) @ displ_vector[[6, 7, 8, 0, 1, 2]]
    p1_31 = np.block([TT1 @ K21_31, TT1 @ K22_31]) @ displ_vector[[6, 7, 8, 0, 1, 2]]

    p1_12 = np.block([TT2 @ K11_12, TT2 @ K12_12]) @ displ_vector[:6]
    p2_12 = np.block([TT2 @ K21_12, TT2 @ K22_12]) @ displ_vector[:6]

    p4_42 = np.block([TT1 @ K11_42, TT1 @ K12_42]) @ displ_vector[[9, 10, 11, 3, 4, 5]]
    p2_42 = np.block([TT1 @ K21_42, TT1 @ K22_42]) @ displ_vector[[9, 10, 11, 3, 4, 5]]

    print("\nNODE INTERNAL FORCES\n")
    print("node 3 bar 3-1:\n", p3_31)
    print("node 1 bar 3-1:\n", p1_31)
    print("node 1 bar 1-2:\n", p1_12)
    print("node 2 bar 1-2:\n", p2_12)
    print("node 2 bar 4-2:\n", p2_42)
    print("node 4 bar 4-2:\n", p4_42)
