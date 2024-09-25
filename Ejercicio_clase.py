import numpy as np
from math import cos, sin, pi

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
    k11 = np.array([[EA/L, 0, 0],
                    [0, 12*EI/pow(L, 3), 6*EI/pow(L, 2)],
                    [0, 6*EI/pow(L, 2), 4*EI/L]
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
    k12 = np.array([[-EA/L, 0, 0],
                    [0, -12*EI/pow(L, 3), 6*EI/pow(L, 2)],
                    [0, -6*EI/pow(L, 2), 2*EI/L]
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
    k21 = np.array([[-EA/L, 0, 0],
                    [0, -12*EI/pow(L, 3), -6*EI/pow(L, 2)],
                    [0, 6*EI/pow(L, 2), 2*EI/L]
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
    k22 = np.array([[EA/L, 0, 0],
                    [0, 12*EI/pow(L, 3), -6*EI/pow(L, 2)],
                    [0, -6*EI/pow(L, 2), 4*EI/L]
                    ])
    glob_coord = T(alpha)
    return glob_coord @ k22 @ np.transpose(glob_coord)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=200)
    #N, mm
    #initial properties
    E = 210E2
    A_IPE300 = 53.80
    I_IPE300 = 8356
    A_HEB160 = 54.25
    I_HEB160 = 2429
    L_viga = 1030
    L_pilar = 463.5
    alpha_pilar1 = pi / 2
    alpha_pilar2 = -pi / 2

    # element properties
    EA_viga = E * A_IPE300
    EI_viga = E * I_IPE300
    EA_pilar = E * A_HEB160
    EI_pilar = E * I_HEB160

    # element matrices
    # first column
    K11_12 = K11(EA_pilar, EI_pilar, L_pilar, alpha_pilar1)
    K12_12 = K12(EA_pilar, EI_pilar, L_pilar, alpha_pilar1)
    K21_12 = K21(EA_pilar, EI_pilar, L_pilar, alpha_pilar1)
    K22_12 = K22(EA_pilar, EI_pilar, L_pilar, alpha_pilar1)

    # beam
    K11_23 = K11(EA_viga, EI_viga, L_viga, 0)
    K12_23 = K12(EA_viga, EI_viga, L_viga, 0)
    K21_23 = K21(EA_viga, EI_viga, L_viga, 0)
    K22_23 = K22(EA_viga, EI_viga, L_viga, 0)

    # first column
    K11_34 = K11(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K12_34 = K12(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K21_34 = K21(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K22_34 = K22(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)


    # zero matrix
    zero = np.zeros((3, 3))
    # stiffness matrix
    stiff_matrix = np.block([
        [K11_12, K12_12, zero, zero],
        [K21_12, K22_12 + K11_23, K12_23, zero],
        [zero, K21_23, K22_23 + K11_34, K12_34],
        [zero, zero, K21_34, K22_34]
    ])

    print("stiffness matrix:\n", stiff_matrix)

    force_vector = np.array([
        1, 1, 1,
        -453.1, 0, -268.64,
        -452.9, -68.75, 428.8,
        1, 1, 0
    ])

    reduced_clm_rows = (3, 4, 5, 6, 7, 8, 11)  # accordint to zeros in displacements vectors
    reduced_stiff_matrix = stiff_matrix[np.ix_(reduced_clm_rows, reduced_clm_rows)]
    print(" reduced stiffness matrix:\n", reduced_stiff_matrix)

    reduced_force_vector = force_vector[[3, 4, 5, 6, 7, 8, 11]]  # flatten the vstack
    print("reduced force vector:\n", reduced_force_vector)

    displacements_reduced = np.linalg.solve(reduced_stiff_matrix, reduced_force_vector)
    print("reduced displacements:\n", displacements_reduced)
