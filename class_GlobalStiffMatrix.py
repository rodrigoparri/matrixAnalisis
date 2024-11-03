import numpy as np
from math import cos, sin
from collections import OrderedDict


class FrameElement:

    def __init__(self, initial_node, end_node, E, A, I, L, alpha, hinged=False):
        """

        :param initial_node: node where local origin is
        :param end_node:
        :param E: normal modulus of elasticity
        :param A: cross-section area
        :param L: element length
        :param alpha: angle between local and global x axes
        """
        self.nodes = (initial_node, end_node)
        self.E = E
        self.A = A
        self.I = I
        self.L = L
        self.alpha = alpha

        self.k_11 = self.K11(E*A, E*I, L, alpha, hinged)
        self.k_12 = self.K12(E*A, E*I, L, alpha, hinged)
        self.k_21 = self.K21(E*A, E*I, L, alpha, hinged)
        self.k_22 = self.K22(E*A, E*I, L, alpha, hinged)
        self.T_transpose = np.transpose(self.T(alpha))

    def __repr__(self):
        return f"""
        nodes: {self.__dict__['nodes']}
        E: {self.__dict__['E']}
        A: {self.__dict__['A']}
        I: {self.__dict__['I']}
        L: {self.__dict__['L']}
        alpha: {self.__dict__['alpha']}
        """

    def T(self, alpha):
        T = np.array([
            [cos(alpha), -sin(alpha), 0],
            [sin(alpha), cos(alpha), 0],
            [0, 0, 1]
        ])
        return T

        # RECORDAR QUE LOS EJES LOCALES POSITIVOS SON X= DERECHA, Y=ARRIBA, M=ANTIHORARIO

    def K11(self, EA, EI, L, alpha, hinged=False):
        """
        :param EA: normal stiffness
        :param EI: bending stiffness
        :param L: bar length
        :param alpha: angle formed between the local x axis and the global x axis
        :return: a numpy matrix
        """
        if hinged==False:
            k11 = np.array([[EA / L, 0, 0],
                            [0, 12 * EI / pow(L, 3), 6 * EI / pow(L, 2)],
                            [0, 6 * EI / pow(L, 2), 4 * EI / L]
                            ])
        else:
            k11 = np.array([[EA / L, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]
                            ])
        glob_coord = self.T(alpha)
        return glob_coord @ k11 @ np.transpose(glob_coord)

    def K12(self, EA, EI, L, alpha, hinged=False):
        """

        :param EA: normal stiffness
        :param EI: bending stiffness
        :param L: bar length
        :param alpha: angle formed between the local x axis and the global x axis
        :return: a numpy matrix
        """
        if hinged==False:
            k12 = np.array([[-EA / L, 0, 0],
                            [0, -12 * EI / pow(L, 3), 6 * EI / pow(L, 2)],
                            [0, -6 * EI / pow(L, 2), 2 * EI / L]
                            ])
        else:
            k12 = np.array([[-EA / L, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]
                            ])
        glob_coord = self.T(alpha)
        return glob_coord @ k12 @ np.transpose(glob_coord)

    def K21(self, EA, EI, L, alpha, hinged=False):
        """

        :param EA: normal stiffness
        :param EI: bending stiffness
        :param L: bar length
        :param alpha: angle formed between the local x axis and the global x axis
        :return: a numpy matrix
        """
        if hinged==False:
            k21 = np.array([[-EA / L, 0, 0],
                            [0, -12 * EI / pow(L, 3), -6 * EI / pow(L, 2)],
                            [0, 6 * EI / pow(L, 2), 2 * EI / L]
                            ])
        else:
            k21 = np.array([[-EA / L, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]
                            ])
        glob_coord = self.T(alpha)
        return glob_coord @ k21 @ np.transpose(glob_coord)

    def K22(self, EA, EI, L, alpha, hinged=False):
        """

        :param EA: normal stiffness
        :param EI: bending stiffness
        :param L: bar length
        :param alpha: angle formed between the local x axis and the global x axis
        :return: a numpy matrix
        """
        if hinged==False:
            k22 = np.array([[EA / L, 0, 0],
                            [0, 12 * EI / pow(L, 3), -6 * EI / pow(L, 2)],
                            [0, -6 * EI / pow(L, 2), 4 * EI / L]
                            ])
        else:
            k22 = np.array([[EA / L, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]
                            ])
        glob_coord = self.T(alpha)
        return glob_coord @ k22 @ np.transpose(glob_coord)


class GlobalMatrix_gen:

    def __init__(self, adjacency: dict, el_propts: dict, force_vector: np.ndarray, displacement_vector: np.ndarray):
        """
        :param adjacency: node adjacency list {0:[0,1], 1:[0,1,2], 2:[1,2]} each node has to include itself
        :param el_propts: element property list {14:{E:1, A: 2, I: 3, L: 4, alpha: 5, hinged: False}, ...}
        element is first node*10 + second node.
        node and element numeration are independent
        """
        np.set_printoptions(threshold=np.inf, linewidth=500)

        self.adjacency = adjacency
        # sort adjacent nodes
        self.node_sorter()
        self.element_properties = el_propts
        self.frame_elements = {}  # {'01': <element>}

        self.create_frame_els()
        self.Global_Matrix = self.global_matrix()
        self.Global_Matrix_str = self.global_matrix_str()

        self.result_displacement_vector = self.solve(force_vector, displacement_vector)
        self.result_internal_forces = self.results()

    def create_frame_els(self):
        already_connected_node_pairs = []
        for k, v in self.adjacency.items():
            for node in v:
                # nodes to connect
                nodes = (min(k, node), max(k, node))
                # check for nodes in already_connected_nodes
                if nodes not in already_connected_node_pairs:
                    element = str(nodes[0]) + str(nodes[1])
                    # create bar element
                    self.frame_elements[element] = FrameElement(
                        nodes[0],
                        nodes[1],
                        self.element_properties[element]['E'],
                        self.element_properties[element]['A'],
                        self.element_properties[element]['I'],
                        self.element_properties[element]['L'],
                        self.element_properties[element]['alpha'],
                        self.element_properties[element]['hinged']
                    )
                    already_connected_node_pairs.append(nodes)
                else:
                    continue

    def initial_row(self):
        """
        :return: list full of zero matrices
        """
        row = []
        for i in self.adjacency:
            row.append(np.zeros((3,3)))

        return row

    def initial_row_str(self):
        row = []
        for i in self.adjacency:
            row.append('0')

        return row

    def isInitialnode(self, i: int, frame_element: str):
        return self.frame_elements[frame_element].nodes[0] == i

    def current_elmnt_str(self, i, j):
        return str(min(i,j)) + str(max(i,j))

    def global_matrix(self):

        row_list = []
        current_element = ''
        for i, nodes in self.adjacency.items():
            current_row = self.initial_row()
            # variable that stores sum of the K matrices of the displacements of node i
            i_K = np.zeros((3, 3))
            # iterate through every node this node is connected to
            for j in nodes:
                # check for symetry in adjacency list
                if i not in self.adjacency[j]:
                    print(f'node {i} not in {j}s adjacency list')
                else:
                    pass
                # set current element to element from i to j
                current_element = self.current_elmnt_str(i, j)
                # check for every of the four possible local matrices
                if self.isInitialnode(j, current_element):
                    current_row[j] = self.frame_elements[current_element].k_21
                    # add matrix of i to the current matrix of i
                    i_K += self.frame_elements[current_element].k_22
                else:
                    current_row[j] = self.frame_elements[current_element].k_12
                    # add matrix of i to the current matrix of i
                    i_K += self.frame_elements[current_element].k_11
            # set i column to the matrix k of the i displacement
            current_row[i] = i_K
            # add current row to global matrix
            row_list.append(np.hstack(current_row))

        global_matrix = np.vstack(row_list)
        return global_matrix

    def global_matrix_str(self):

        row_list = []
        current_element = ''
        for i, nodes in self.adjacency.items():
            # reset current row
            current_row = self.initial_row_str()
            # variable that stores sum of the K matrices of the displacements of node i
            i_K = ''
            # iterate through every node this node is connected to
            for j in nodes:
                # set current element to element from i to j
                current_element = self.current_elmnt_str(i, j)
                # check for every of the four possible local matrices
                if self.isInitialnode(j, current_element):
                    current_row[j] = f'K21_{current_element}'
                    # add matrix of i to the current matrix of i
                    i_K += f'K22_{current_element}'
                else:
                    current_row[j] = f'K12_{current_element}'
                    # add matrix of i to the current matrix of i
                    i_K += f'K11_{current_element}'
            # set i column to the matrix k of the i displacement
            current_row[i] = i_K
            # add current row to global matrix
            row_list.append(np.hstack(current_row))

        global_matrix = np.vstack(row_list)

        return global_matrix

    def node_sorter(self):
        """
        sorts adjacent nodes from smaller to bigger
        :return: sorted adjacency list
        """
        ordered_adjacency = OrderedDict(sorted(self.adjacency.items()))
        self.adjacency = ordered_adjacency
        for node in self.adjacency:
            self.adjacency[node].sort()

    def get_reduced_clm_row(self, displacement_vector):
        """
        :param displacement_vector: np.array with 0 if global displacement of the is 0 and 1 if is an unknown value
        :return: list of indexes where diplacement vector is non zero
        """
        reduced_clm_row = []
        for i in range(0, len(displacement_vector) - 1):
            if displacement_vector[i] != 0:
                reduced_clm_row.append(i)
            else:
                continue

        return reduced_clm_row

    def get_reduced_matrix(self, reduced_clm_row):
        reduced_matrix = self.Global_Matrix[np.ix_(reduced_clm_row, reduced_clm_row)]
        return reduced_matrix

    def get_reduced_vector(self, vector, reduced_clm_row):
        reduced_vector = vector[reduced_clm_row]
        return reduced_vector

    def solve(self, force_vector, displacement_vector):
        reduced_clm_row = self.get_reduced_clm_row(displacement_vector)
        reduced_stiff_matrix = self.get_reduced_matrix(reduced_clm_row)
        reduced_force_vector = self.get_reduced_vector(force_vector, reduced_clm_row)
        reduced_displacement_vector = np.linalg.solve(reduced_stiff_matrix, reduced_force_vector)

        # complete displacement vector with results
        disp_list = []
        # current item in reduced vector index
        red_vect_index = 0
        for i in range(0, len(force_vector)):
            # if index is in extracted rows and columns
            if i in reduced_clm_row:
                disp_list.append(reduced_displacement_vector[red_vect_index])
                red_vect_index += 1
            else:
                disp_list.append(0)

        result_displacement_vector = np.array(disp_list)

        return result_displacement_vector

    def results(self):
        # internal forces in local coordinates of each element
        elements_internal_forces = {}
        for name, element in self.frame_elements.items():

            # names of nodes force vectors
            name_node1 = f'p{element.nodes[0]}_{name}'
            name_node2 = f'p{element.nodes[1]}_{name}'

            bounds = [(3 * element.nodes[0], 3 * element.nodes[0] + 2), (3 * element.nodes[1], 3 * element.nodes[1] + 2)]
            # global displacements only concerning considered nodes
            displacement_vector = np.concatenate([self.result_displacement_vector[start:end+1] for start, end in bounds])
            elements_internal_forces[name_node1] = np.block([element.T_transpose @ element.k_11, element.T_transpose @ element.k_12]) @ displacement_vector
            elements_internal_forces[name_node2] = np.block([element.T_transpose @ element.k_21, element.T_transpose @ element.k_22]) @ displacement_vector

        return  elements_internal_forces

if __name__ == '__main__':
    adjacency = {0: [1],
                 1: [0, 2, 4],
                 2: [1, 3],
                 3: [2, 4],
                 4: [3, 1, 5],
                 5: [4]
                 }

    el_propts = {
        '01': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '12': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '23': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '34': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '45': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False},
        '14': {'E': 1, 'A': 2, 'I': 3, 'L': 4, 'alpha': 5, 'hinged': False}
    }

