import numpy as np

from class_GlobalStiffMatrix import GlobalMatrix_gen

if __name__ == '__main__':
    adjacency = {0: [1],
                 1: [0, 2, 4],
                 2: [1, 3],
                 3: [2, 4],
                 4: [3, 1, 5],
                 5: [4]
                 }

    element_properties = {
        '01': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '12': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '23': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '34': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '45': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '14': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False}
    }

    force_vector = np.array([
        1, 1, 1,
        0, -370, -320,
        0, -490, -200,
        -40, -490, 200,
        -40, -370, 320,
        1, 1, 1
    ])

    displacement_vector = np.array([
        0, 0, 0,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        0, 0, 0
    ])

    Matrix_gen = GlobalMatrix_gen(adjacency, element_properties, force_vector, displacement_vector)
    Global_matrix = Matrix_gen.Global_Matrix
    Global_matrix_str = Matrix_gen.Global_Matrix_str

    for node, forces in Matrix_gen.result_internal_forces.items():
        print(f'{node}: {forces}', '\n')
