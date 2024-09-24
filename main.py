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

def print_block_matrix(block_matrix, submatrix_size=3):
    rows, cols = block_matrix.shape
    # Iterate through the matrix row-wise, printing each submatrix block
    for i in range(0, rows, submatrix_size):
        for j in range(submatrix_size):  # For each row in the submatrix
            row_str = ""
            for k in range(0, cols, submatrix_size):  # For each submatrix column block
                row_str += " ".join(f"{block_matrix[i + j, k + l]:2}" for l in range(submatrix_size)) + "    "
            print(row_str)
        print("")  # Blank line between block rows

if __name__ == '__main__':
    #N, mm
    #initial properties
    E = 210E3
    A_IPE300 = 5380
    I_IPE300 = 8356E4
    A_HEB160 = 5425
    I_HEB160 = 2429E4
    L_viga = 5.8
    L_pilar = 2.9
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

    # beam first half
    K11_23 = K11(EA_viga, EI_viga, L_viga, 0)
    K12_23 = K12(EA_viga, EI_viga, L_viga, 0)
    K21_23 = K21(EA_viga, EI_viga, L_viga, 0)
    K22_23 = K22(EA_viga, EI_viga, L_viga, 0)

    # beam second half
    K11_34 = K11(EA_viga, EI_viga, L_viga, 0)
    K12_34 = K12(EA_viga, EI_viga, L_viga, 0)
    K21_34 = K21(EA_viga, EI_viga, L_viga, 0)
    K22_34 = K22(EA_viga, EI_viga, L_viga, 0)

    # first column
    K11_45 = K11(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K12_45 = K12(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K21_45 = K21(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K22_45 = K22(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)

    stiff_matrix = np.block([
        [K11_12, K12_12, 0, 0, 0,],
        [K21_12, K22_12 + K11_23, K12_23, 0, 0],
        [0, K21_23, K22_23 + K11_34, K12_34, 0],
        [0, 0, K21_34, K22_34 + K11_45, K12_45],
        [0, 0, 0, K21_45, K22_45]
    ])

