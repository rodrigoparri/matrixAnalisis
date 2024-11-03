import numpy as np

from class_GlobalStiffMatrix import GlobalMatrix_gen

if __name__ == '__main__':
    adjacency = {
        0: [1, 2],
        1: [0,3],
        2: [0],
        3: [1]
                 }

    element_properties = {
        '01': {'E': 200E9, 'A': 2.5E-2, 'I': 5E-4, 'L': 5, 'alpha': 0, 'hinged': False},
        '02': {'E': 200E9, 'A': 2.5E-2, 'I': 5E-4, 'L': 4, 'alpha': np.pi / 2, 'hinged': False},
        '13': {'E': 200E9, 'A': 2.5E-2, 'I': 5E-4, 'L': 4, 'alpha': np.pi / 2, 'hinged': False},
    }

    force_vector = np.array([
        0, 0, 1E4,
        1, 0, 20E3,
        1, 1, 1,
        1, 1, 1,
    ])

    displacement_vector = np.array([
        1, 1, 1,
        0, 1, 1,
        0, 0, 0,
        0, 0, 0
    ])

    Matrix_gen = GlobalMatrix_gen(adjacency, element_properties, force_vector, displacement_vector)
    Global_matrix = Matrix_gen.Global_Matrix
    Global_matrix_str = Matrix_gen.Global_Matrix_str

    for node, forces in Matrix_gen.result_internal_forces.items():
        print(f'{node}: {forces}', '\n')
