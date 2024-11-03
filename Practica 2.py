import numpy as np
from math import cos, sin, pi, sqrt, radians

def T(alpha):
    T = np.array([
        [cos(alpha), -sin(alpha), 0],
        [sin(alpha), cos(alpha), 0],
        [0, 0, 1]
                ])
    return T

# RECORDAR QUE LOS EJES LOCALES POSITIVOS SON X= DERECHA, Y=ARRIBA, M=ANTIHORARIO
def K11(EA, EI, L, alpha, hinged=False):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    if hinged == False:
        k11 = np.array([[EA/L, 0, 0],
                    [0, 12*EI/pow(L, 3), 6*EI/pow(L, 2)],
                    [0, 6*EI/pow(L, 2), 4*EI/L]
                    ])
    else:
        k11 = np.array([[EA/L, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]
                        ])
    glob_coord = T(alpha)
    return glob_coord @ k11 @ np.transpose(glob_coord)

def K12(EA, EI, L, alpha, hinged = False):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    if hinged == False:
        k12 = np.array([[-EA/L, 0, 0],
                        [0, -12*EI/pow(L, 3), 6*EI/pow(L, 2)],
                        [0, -6*EI/pow(L, 2), 2*EI/L]
                        ])
    else:
        k12 = np.array([[-EA/L, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]
                        ])
    glob_coord = T(alpha)
    return glob_coord @ k12 @ np.transpose(glob_coord)

def K21(EA, EI, L, alpha, hinged = False):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    if hinged == False:
        k21 = np.array([[-EA/L, 0, 0],
                        [0, -12*EI/pow(L, 3), -6*EI/pow(L, 2)],
                        [0, 6*EI/pow(L, 2), 2*EI/L]
                        ])
    else:
        k21 = np.array([[-EA / L, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]
                        ])
    glob_coord = T(alpha)
    return glob_coord @ k21 @ np.transpose(glob_coord)

def K22(EA, EI, L, alpha, hinged = False):
    """

    :param EA: normal stiffness
    :param EI: bending stiffness
    :param L: bar length
    :param alpha: angle formed between the local x axis and the global x axis
    :return: a numpy matrix
    """
    if hinged == False:
        k22 = np.array([[EA/L, 0, 0],
                        [0, 12*EI/pow(L, 3), -6*EI/pow(L, 2)],
                        [0, -6*EI/pow(L, 2), 4*EI/L]
                        ])
    else:
        k22 = np.array([[EA/L, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]
                        ])
    glob_coord = T(alpha)
    return glob_coord @ k22 @ np.transpose(glob_coord)

def isSimetric(matrix:np.array):
    return np.array_equal(matrix,np.transpose(matrix))

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, linewidth=500)
    # kN, m
    # initial properties
    E = 210E6
    A_IPE500 = 115.5E-4
    I_IPE500 = 48200E-8  # twice the actual ipe400 inertia
    A_HEB300 = 149.1E-4
    I_HEB300 = 25170E-8
    A_HEB320 = 161.30E-4
    I_HEB320 = 30820E-8
    A_struts = 19.2E-4
    L_12 = 4
    L_23 = 8
    L_35 = 6
    L_13 = sqrt(L_12 ** 2 + L_23 ** 2)

    alpha_12 = pi / 2
    alpha_23 = 0
    alpha_34 = -pi / 2
    alpha_35 = 0
    alpha_13 = radians(26.57)
    alpha_24 = -alpha_13

    # element properties
    EA_viga = E * A_IPE500
    EI_viga = E * I_IPE500
    EA_pilar_12 = E * A_HEB300
    EI_pilar_12 = E * I_HEB300
    EA_pilar_34 = E * A_HEB320
    EI_pilar_34 = E * I_HEB320
    EA_struts = E * A_struts

    # element matrices
    # first column
    K11_12 = K11(EA_pilar_12, EI_pilar_12, L_12, alpha_12)
    K12_12 = K12(EA_pilar_12, EI_pilar_12, L_12, alpha_12)
    K21_12 = K21(EA_pilar_12, EI_pilar_12, L_12, alpha_12)
    K22_12 = K22(EA_pilar_12, EI_pilar_12, L_12, alpha_12)

    # beam 23
    K11_23 = K11(EA_viga, EI_viga, L_23, 0)
    K12_23 = K12(EA_viga, EI_viga, L_23, 0)
    K21_23 = K21(EA_viga, EI_viga, L_23, 0)
    K22_23 = K22(EA_viga, EI_viga, L_23, 0)

    # second column
    K11_34 = K11(EA_pilar_34, EI_pilar_34, L_12, alpha_34)
    K12_34 = K12(EA_pilar_34, EI_pilar_34, L_12, alpha_34)
    K21_34 = K21(EA_pilar_34, EI_pilar_34, L_12, alpha_34)
    K22_34 = K22(EA_pilar_34, EI_pilar_34, L_12, alpha_34)

    # second beam half
    K11_35 = K11(EA_viga, EI_viga, L_35, alpha_35)
    K12_35 = K12(EA_viga, EI_viga, L_35, alpha_35)
    K21_35 = K21(EA_viga, EI_viga, L_35, alpha_35)
    K22_35 = K22(EA_viga, EI_viga, L_35, alpha_35)

    # strut 13
    K11_13 = K11(EA_struts, 0, L_13, alpha_13, True)
    K12_13 = K12(EA_struts, 0, L_13, alpha_13, True)
    K21_13 = K21(EA_struts, 0, L_13, alpha_13, True)
    K22_13 = K22(EA_struts, 0, L_13, alpha_13, True)

    # strut 24
    K11_24 = K11(EA_struts, 0, L_13, alpha_24, True)
    K12_24 = K12(EA_struts, 0, L_13, alpha_24, True)
    K21_24 = K21(EA_struts, 0, L_13, alpha_24, True)
    K22_24 = K22(EA_struts, 0, L_13, alpha_24, True)

    # zero matrix
    zero = np.zeros((3, 3))
    stiff_matrix = np.block([
        [K11_12 + K11_13, K12_12, K12_13, zero, zero],
        [K21_12, K22_12 + K11_23 + K11_24, K12_23, K12_24, zero],
        [K21_13, K21_23, K22_23 + K22_13 + K11_34 + K11_35, K12_34, K12_35],
        [zero, K21_24, K21_34, K22_34, zero],
        [zero, zero, K21_35, zero, K22_35]
    ])

    print("stiffness matrix:\n", stiff_matrix, f"\n Stiffness matrix is simetric: {isSimetric(stiff_matrix)}")

    force_vector = np.array([
        1, 1, 1,
        150, -424, -162.1,
        0, -510, 33.3,
        1, 1, 0,
        0, -154, -51.2
    ])

    reduced_clm_rows = [3, 4, 5, 6, 7, 8, 11, 12, 14]  # according to non zeros in displacements vectors by index
    reduced_stiff_matrix = stiff_matrix[np.ix_(reduced_clm_rows, reduced_clm_rows)]
    print(" reduced stiffness matrix:\n", reduced_stiff_matrix)

    reduced_force_vector = force_vector[reduced_clm_rows]
    print("reduced force vector: (global coordinates)\n", reduced_force_vector)

    reduced_displ_vector = np.linalg.solve(reduced_stiff_matrix, reduced_force_vector)
    print("reduced displacements(global coordinates):\n", reduced_displ_vector)

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
    print("displacement vector(global coordinates):\n", displ_vector)

    result_force_vector = stiff_matrix @ displ_vector
    print("result force vector(global coordinates):\n", result_force_vector)

    # inner forces in each element
    # T transpose for first column
    TT1 = np.transpose(T(alpha_12))
    # T transpose for beam
    TT2 = np.transpose(T(0))
    # T transpose for second column
    TT3 = np.transpose(T(alpha_34))
    # T transpose for first strut
    TT4 = np.transpose(T(alpha_13))
    # T transpose for second strut
    TT5 = np.transpose(T(alpha_24))

    # internal forces in each bar element
    p1_12 = np.block([TT1 @ K11_12, TT1 @ K12_12]) @ displ_vector[:6]
    p2_12 = np.block([TT1 @ K21_12, TT1 @ K22_12]) @ displ_vector[:6]

    print(displ_vector[3:9])
    p2_23 = np.block([TT2 @ K11_23, TT2 @ K12_23]) @ displ_vector[3:9]
    p3_23 = np.block([TT2 @ K21_23, TT2 @ K22_23]) @ displ_vector[3:9]

    p3_34 = np.block([TT3 @ K11_34, TT3 @ K12_34]) @ displ_vector[6:12]
    p4_34 = np.block([TT3 @ K21_34, TT3 @ K22_34]) @ displ_vector[6:12]

    p3_35 = np.block([TT2 @ K11_35, TT2 @ K12_35]) @ displ_vector[[6,7,8,12,13,14]]
    p5_35 = np.block([TT2 @ K21_35, TT2 @ K22_35]) @ displ_vector[[6,7,8,12,13,14]]

    p1_13 = np.block([TT4 @ K11_13, TT4 @ K12_13]) @ displ_vector[[0,1,2,6,7,8]]
    p3_13 = np.block([TT4 @ K21_13, TT4 @ K22_13]) @ displ_vector[[0,1,2,6,7,8]]

    p2_24 = np.block([TT5 @ K11_24, TT5 @ K12_24]) @ displ_vector[[3,4,5,9,10,11]]
    p4_24 = np.block([TT5 @ K21_24, TT5 @ K22_24]) @ displ_vector[[3,4,5,9,10,11]]

    print("\nNODE INTERNAL FORCES (local coordinates)\n")
    print("node 1 bar 1-2:\n", p1_12)
    print("node 2 bar 1-2:\n", p2_12)
    print("node 2 bar 2-3:\n", p2_23)
    print("node 3 bar 2-3:\n", p3_23)
    print("node 3 bar 3-4:\n", p3_34)
    print("node 4 bar 3-4:\n", p4_34)
    print("node 3 bar 3-5:\n", p3_35)
    print("node 5 bar 3-5:\n", p5_35)
    print('node 1 strut 1-3: \n', p1_13)
    print('node 3 strut 1-3: \n', p3_13)
    print('node 2 strut 2-4: \n', p2_24)
    print('node 4 strut 2-4: \n', p4_24)
