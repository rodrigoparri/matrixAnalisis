import numpy as np
from math import cos, sin


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

    def __init__(self, adjacency: dict, el_propts: dict):
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
        self.el_propts = el_propts
        self.frame_els = {}  # {'01': <element>}

        self.create_frame_els()
        self.global_matrix()
        self.global_matrix_str()
        #print(self.frame_els['01'].k_11)

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
                    self.frame_els[element] = FrameElement(
                        nodes[0],
                        nodes[1],
                        self.el_propts[element]['E'],
                        self.el_propts[element]['A'],
                        self.el_propts[element]['I'],
                        self.el_propts[element]['L'],
                        self.el_propts[element]['alpha'],
                        self.el_propts[element]['hinged']
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
        return self.frame_els[frame_element].nodes[0] == i

    def current_elmnt_str(self, i, j):
        return str(min(i,j)) + str(max(i,j))

    def global_matrix(self):

        row_list = []
        current_element = ''
        for i, nodes in adjacency.items():
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
                    current_row[j] = self.frame_els[current_element].k_21
                    # add matrix of i to the current matrix of i
                    i_K += self.frame_els[current_element].k_22
                else:
                    current_row[j] = self.frame_els[current_element].k_12
                    # add matrix of i to the current matrix of i
                    i_K += self.frame_els[current_element].k_11
            # set i column to the matrix k of the i displacement
            current_row[i] = i_K
            # add current row to global matrix
            row_list.append(np.hstack(current_row))

        global_matrix = np.vstack(row_list)
        print(global_matrix)
        return global_matrix

    def global_matrix_str(self):

        row_list = []
        current_element = ''
        for i, nodes in adjacency.items():
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

        print(global_matrix)
        return global_matrix

    def node_sorter(self):
        """
        sorts adjacent nodes from smaller to bigger
        :return: sorted adjacency list
        """
        for node in self.adjacency:
            self.adjacency[node].sort()


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

    matrix_gen = GlobalMatrix_gen(adjacency, el_propts)
