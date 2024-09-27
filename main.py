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
    np.set_printoptions(threshold=np.inf, linewidth=500)
    #kN, m
    #initial properties
    E = 210E6
    A_IPE300 = 53.8E-4
    I_IPE300 = 8356E-8
    A_HEB160 = 54.25E-4
    I_HEB160 = 2429E-8
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

    # second column
    K11_45 = K11(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K12_45 = K12(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K21_45 = K21(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)
    K22_45 = K22(EA_pilar, EI_pilar, L_pilar, alpha_pilar2)

    # zero matrix
    zero = np.zeros((3, 3))
    stiff_matrix = np.block([
        [K11_12, K12_12, zero, zero, zero],
        [K21_12, K22_12 + K11_23, K12_23, zero, zero],
        [zero, K21_23, K22_23 + K11_34, K12_34, zero],
        [zero, zero, K21_34, K22_34 + K11_45, K12_45],
        [zero, zero, zero, K21_45, K22_45]
    ])

    print("stiffness matrix:\n", stiff_matrix)

    force_vector = np.array([
        1, 1, 1,
        20, -357.5, -66.33,
        20, -435, 0,
        20, -377.5, 46.93,
        1, 1, 1
    ])

    reduced_clm_rows = [3, 4, 5, 6, 7, 8, 9, 10, 11]  # according to zeros in displacements vectors
    reduced_stiff_matrix = stiff_matrix[np.ix_(reduced_clm_rows, reduced_clm_rows)]
    print(" reduced stiffness matrix:\n", reduced_stiff_matrix)

    reduced_force_vector = force_vector[reduced_clm_rows]
    print("reduced force vector:\n", reduced_force_vector)

    reduced_displ_vector = np.linalg.solve(reduced_stiff_matrix, reduced_force_vector)
    print("reduced displacements:\n", reduced_displ_vector)

    # future displacement vector
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
    TT1 = np.transpose(T(alpha_pilar1))
    # T transpose for beam
    TT2 = np.transpose(T(0))
    # T transpose for second column
    TT3 = np.transpose(T(alpha_pilar2))

    # internal forces in each barr element
    p1_12 = np.block([TT1 @ K11_12, TT1 @ K12_12]) @ displ_vector[:6]
    p2_12 = np.block([TT1 @ K21_12, TT1 @ K22_12]) @ displ_vector[:6]

    p2_23 = np.block([TT2 @ K11_23, TT2 @ K12_23]) @ displ_vector[3:9]
    p3_23 = np.block([TT2 @ K21_23, TT2 @ K22_23]) @ displ_vector[3:9]

    p3_34 = np.block([TT2 @ K11_34, TT2 @ K12_34]) @ displ_vector[6:12]
    p4_34 = np.block([TT2 @ K21_34, TT2 @ K22_34]) @ displ_vector[6:12]

    p4_45 = np.block([TT3 @ K11_45, TT3 @ K12_45]) @ displ_vector[9:]
    p5_45 = np.block([TT3 @ K21_45, TT3 @ K22_45]) @ displ_vector[9:]

    print("\nNODE INTERNAL FORCES\n")
    print("node 1 bar 1-2:\n", p1_12)
    print("node 2 bar 1-2:\n", p2_12)
    print("node 2 bar 2-3:\n", p2_23)
    print("node 3 bar 2-3:\n", p3_23)
    print("node 3 bar 3-4:\n", p3_34)
    print("node 4 bar 3-4:\n", p4_34)
    print("node 4 bar 4-5:\n", p4_45)
    print("node 5 bar 4-5:\n", p5_45)
