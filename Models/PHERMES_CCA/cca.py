from sklearn.preprocessing import MinMaxScaler
from phermes import HyperHeuristic
from sklearn import linear_model
from phermes import Problem
from typing import List
import pandas as pd
import numpy as np

def find_cell_index(i, x_i, data_min, data_max, m):
    """
    Finds the cell index in the CA grid where a data point should be mapped.
    It locates the corresponding cell where a data point fits within the cell range.
    Receives:
        i: Attribute index
        x_i: Field or attribute value
        data_min: Minimum value of all record for each attribute
        data_max: Maximum value of all record for each attribute
        m: Amount of cells in CA (one-row)
    Returns:
        index: An index value that corresponds to the CA cell where the data instance value matches
    """
    numerator = x_i - data_min[i]
    denominator_p1 = data_max[i] - data_min[i]
    denominator_p2 = m - 1
    denominator_c = denominator_p1/denominator_p2 
    index = int(np.floor(numerator / denominator_c))
    # Guardian clip for index values that could exceed the CA boundaries(0 - m - 1)
    return np.clip(index, 0, m-1)
def map_data_to_CA_cells(data_instance, n_attributes, label, data_min, data_max, CA, m):
    """
    Maps data points to cells in the CA grid, a cell receives a positive count value when the data point matched the cell.
    Receives:
        data_instance: Values of all attributes or features associated with an specific record
        n_attributes: Amount of attributes of each data instance
        label: Class or label assigned to a data instance
        data_min: Minimum value of all records for each attribute
        data_max: Maximum value of all records for each attribute
        m: Amount of cells in CA grid (one-row)
    Returns:
        CA: Updated CA with the frequency or count of real data points that were mapped in each cell
    """
    for i in range(n_attributes):
        index = find_cell_index(i, data_instance[i], data_min, data_max, m)
        CA[label][i][index] += 1
    return CA
def set_temperature_values(A, n_attributes, m):
    """
    Sets the temperature values in the CA grid.
    Transforms the counts or the frequency into a logarithmic scaled value.
    Receives:
        A: Single CA grid (one-row)
        n_attributes: Amount of attributes of each data instance
        m: Amount of cells in CA grid (one-row)
    Returns:
        A: Updated single CA grid (one-row) with the converted states, counts or frequencies
    """
    for i in range(n_attributes):
        for j in range(m):
            A[i][j] = np.log(A[i][j] + 1)
    return A
# def distribute_heat(A, n_attributes, m, portion, range_param):
#     """
#     Distributes the heat in the CA grid.
#     The different counts are the temperature containg a driving force or heat which is transfered according to some defined rules.
#     In other words, it applies some rules that distribute the cells count values.
#     1. Each cell that has a temperature value emits a heat wave in two directions (left & right)
#     2. Heat waves spread to the neighbor cells until they reach the borders of the area determined by the range parameter (percentage of cells that will be utilized in the heat transfer process)
#     3. Cells that received energy, emit a portion of the energy they have received, into the direction of the original wave (only)
#     4. A cell is not supposed to emit more than one heat wave, it can only emit it one

#     Receive:
#         A: Single CA grid (one-row)
#         n_attributes: Amount of attributes of each data instance
#         m: Amount of cells in CA grid (one-row)
#         portion: Percentage of heat that will be transfered
#         range_param: Percentage of cells that will be utilized in the heat transfer process.
#     Return:
#         A: Updated single CA grid (one-row) with the heat distributed, in other words the counts reaccomodated
#     """
#     for attribute_index in range(n_attributes):
#         index_lst = [[0, 0, 0] for _ in range(m)]
#         new_row = [0] * m
#         for index in range(m):
#             if A[attribute_index][index] != 0:
#                 if index == 0:
#                     index_lst[index][0] = 1
#                     index_lst[index + range_param][2] = 1
#                     new_row[1] += A[attribute_index][index] * portion
#                     new_row[0] += A[attribute_index][index] * (1 - portion) 
#                 elif index == m - 1:
#                     index_lst[index][0] = 1
#                     index_lst[index - range_param][1] = 1
#                     new_row[m - 2] += A[attribute_index][index] * portion
#                     new_row[m - 1] += A[attribute_index][m - 1] * (1 - portion)
#                 else:
#                     index_lst[index][0] = 1
#                     index_lst[index + range_param][2] = 1
#                     index_lst[index - range_param][1] = 1
#                     left_index = index - range_param
#                     right_index = index + range_param
#                     new_row[left_index] += A[attribute_index][index] * portion/2
#                     new_row[right_index] += A[attribute_index][index] * portion/2
#                     new_row[index] += A[attribute_index][index] * (1 - portion)
#         new_row_2 = [0] * m
#         for index in range(len(index_lst)):
#             if index_lst[index][0] == 0:
#                 if index != 0 and index != m - 1:
#                     if index_lst[index][1] and index_lst[index][2]:
#                         index_lst[index][0] = 1
#                         index_lst[index - range_param][1] = 1
#                         index_lst[index + range_param][2] = 1
#                         left_index = index - range_param
#                         right_index = index + range_param
#                         if new_row[left_index] != 0 and new_row[right_index] != 0:
#                             new_row_2[left_index] = new_row[left_index] + new_row[index] * portion/2
#                             new_row_2[right_index] = new_row[right_index] + new_row[index] * portion/2
#                             new_row_2[index] = new_row[index] * (1 - portion)
#                         elif new_row[left_index] != 0:
#                             new_row_2[left_index] = new_row[left_index] + new_row[index] * portion/2
#                             new_row_2[right_index] = new_row[right_index] * portion/2
#                             new_row_2[index] = new_row[index] * (1 - portion)
#                         elif new_row[right_index] != 0:
#                             new_row_2[left_index] = new_row[index] * portion/2
#                             new_row_2[right_index] = new_row[right_index] + new_row[index] * portion/2
#                             new_row_2[index] = new_row[index] * (1 - portion)                    
#                         else:
#                             new_row_2[left_index] = new_row[index] * portion/2
#                             new_row_2[right_index] = new_row[index] * portion/2
#                             new_row_2[index] = new_row[index] * (1 - portion)                    
#                     elif index_lst[index][1]:
#                         index_lst[index][0] = 1
#                         index_lst[index - range_param][1] = 1                
#                         left_index = index - range_param
#                         if new_row[left_index] != 0:
#                             new_row_2[left_index] = new_row[index] + new_row[index] * portion
#                             new_row_2[index] = new_row[index] * (1 - portion)   
#                         else:
#                             new_row_2[left_index] = new_row[index] * portion
#                             new_row_2[index] = new_row[index] * (1 - portion)                
#                     elif index_lst[index][2]:
#                         index_lst[index][0] = 1
#                         index_lst[index + range_param][2] = 1                
#                         right_index = index + range_param                
#                         if new_row[right_index] != 0:
#                             new_row_2[right_index] = new_row[index] + new_row[index] * portion
#                             new_row_2[index] = new_row[index] * (1 - portion)
#                         else:
#                             new_row_2[right_index] = new_row[index] * portion
#                             new_row_2[index] = new_row[index] * (1 - portion)
#             else:
#                 if new_row_2[index] == 0:
#                     new_row_2[index] = new_row[index]
#         A[attribute_index] = new_row_2
#     return A
# def distribute_heat(A, n_attributes, m, portion, range_percentage):
#     def transfer_heat(A, n_attributes, m, portion, range_percentage):
#         index_dict = {}
#         for i in range(n_attributes):
#             new_row = np.zeros(m)
#             oside_neighborhood = int(m*range_percentage)
#             ind_portion = portion/(2*oside_neighborhood)
#             index_lst = [[0, 0, 0] for _ in range(m)]
#             for j in range(m):
#                 if A[i][j] > 0:
#                     if j - oside_neighborhood < 0:
#                         left_index = max(0, j - oside_neighborhood)
#                         right_index = min(m - 1, j + oside_neighborhood)
#                         ind_portion_adj = portion/(2*oside_neighborhood - abs(j - oside_neighborhood))
#                         new_row[j + 1:right_index + 1] += A[i][j] * ind_portion_adj
#                         new_row[left_index:j] += A[i][j] * ind_portion_adj
#                         new_row[j] += A[i][j] * (1-portion)
#                         index_lst[j][0] = 1
#                         for sublst in index_lst[j + 1:right_index + 1]:
#                             sublst[1] = 1
#                         for sublst in index_lst[left_index:j]:
#                             sublst[2] = 1
#                     elif j + oside_neighborhood > m - 1:
#                         left_index = max(0, j - oside_neighborhood)
#                         right_index = min(m - 1, j + oside_neighborhood)
#                         ind_portion_adj = portion/(2*oside_neighborhood - abs(m - 1 - (j + oside_neighborhood)))
#                         new_row[j + 1:right_index + 1] += A[i][j] * ind_portion_adj
#                         new_row[left_index:j] += A[i][j] * ind_portion_adj
#                         new_row[j] += A[i][j] * (1-portion)
#                         index_lst[j][0] = 1
#                         for sublst in index_lst[j + 1:right_index + 1]:
#                             sublst[1] = 1
#                         for sublst in index_lst[left_index:j]:
#                             sublst[2] = 1
#                     elif j == 0:
#                         right_index = min(m - 1, j + oside_neighborhood)
#                         ind_portion_adj = portion/oside_neighborhood
#                         new_row[1:right_index + 1] += A[i][j] * ind_portion_adj
#                         new_row[j] += A[i][j] * (1 - poriton)
#                         index_lst[j][0] = 1
#                         for sublst in index_lst[1:right_index + 1]:
#                             sublst[1] = 1
#                     elif j == m - 1:
#                         left_index = max(0, j - oside_neighborhood)
#                         ind_portion_adj = portion/oside_neighborhood
#                         new_row[left_index:m - 1] += A[i][j] * ind_portion_adj
#                         new_row[j] += A[i][j] * (1 - poriton)
#                         index_lst[j][0] = 1
#                         for sublst in index_lst[left_index:m - 1]:
#                             sublst[2] = 1
#                     else:
#                         left_index = max(0, j - oside_neighborhood)
#                         right_index = min(m - 1, j + oside_neighborhood)
#                         new_row[j + 1:right_index + 1] += A[i][j] * ind_portion
#                         new_row[left_index:j] += A[i][j] * ind_portion
#                         new_row[j] += A[i][j] * (1-portion)
#                         index_lst[j][0] = 1
#                         for sublst in index_lst[j + 1:right_index + 1]:
#                             sublst[1] = 1
#                         for sublst in index_lst[left_index:j]:
#                             sublst[2] = 1
#             index_dict[i] = index_lst        
#             A[i] = new_row
#         return A, index_dict
#     def expand_heat(A, n_attributes, m, portion, range_percentage, index_dict):
#         updated_index_dict = {}
#         for i in range(n_attributes):
#             new_row = np.zeros(m)
#             oside_neighborhood = int(m*range_percentage)
#             ind_portion = portion/(2*oside_neighborhood)
#             index_lst = index_dict[i].copy()
#             skipped = []
#             for j in range(m):
#                 if j != 0 and j != m - 1:
#                     if index_dict[i][j][0] == 0: 
#                         if index_dict[i][j][1] == 1 and index_dict[i][j][2] == 0:
#                             if j + oside_neighborhood > m - 1:
#                                 right_index = min(m - 1, j + oside_neighborhood)
#                                 ind_portion_adj = portion/(oside_neighborhood - abs(m - 1 - (j + oside_neighborhood)))
#                                 new_row[j + 1:right_index + 1] += A[i][j + 1:right_index + 1] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[j + 1:right_index + 1]:
#                                     sublst[1] = 1
#                             else:
#                                 right_index = j + oside_neighborhood
#                                 ind_portion_adj = portion/oside_neighborhood
#                                 new_row[j + 1:right_index + 1] += A[i][j + 1:right_index + 1] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[j + 1:right_index + 1]:
#                                     sublst[1] = 1
#                         elif index_dict[i][j][1] == 0 and index_dict[i][j][2] == 1:
#                             if j - oside_neighborhood < 0:
#                                 left_index = max(0, j - oside_neighborhood)
#                                 ind_portion_adj = portion/(oside_neighborhood - abs(j - oside_neighborhood))
#                                 new_row[left_index:j] += A[i][left_index:j] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[left_index:j]:
#                                     sublst[2] = 1
#                             else:
#                                 left_index = j - oside_neighborhood
#                                 ind_portion_adj = portion/oside_neighborhood
#                                 new_row[left_index:j] += A[i][left_index:j] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[left_index:j]:
#                                     sublst[2] = 1
#                         elif index_dict[i][j][1] == 1 and index_dict[i][j][2] == 1:
#                             if j + oside_neighborhood > m - 1:
#                                 left_index = max(0, j - oside_neighborhood)
#                                 right_index = min(m - 1, j + oside_neighborhood)
#                                 ind_portion_adj = portion/(2*oside_neighborhood - abs(m - 1 - (j + oside_neighborhood)))
#                                 new_row[j + 1:right_index + 1] += A[i][j + 1:right_index + 1] + A[i][j] * ind_portion_adj
#                                 new_row[left_index:j] += A[i][left_index:j] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[j + 1:right_index + 1]:
#                                     sublst[1] = 1
#                                 for sublst in index_lst[left_index:j]:
#                                     sublst[2] = 1
#                             if j - oside_neighborhood < 0:
#                                 left_index = max(0, j - oside_neighborhood)
#                                 right_index = min(m - 1, j + oside_neighborhood)
#                                 ind_portion_adj = portion/(2*oside_neighborhood - abs(j - oside_neighborhood))
#                                 new_row[j + 1:right_index + 1] += A[i][j + 1:right_index + 1] + A[i][j] * ind_portion_adj
#                                 new_row[left_index:j] += A[i][left_index:j] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[j + 1:right_index + 1]:
#                                     sublst[1] = 1
#                                 for sublst in index_lst[left_index:j]:
#                                     sublst[2] = 1
#                             else:
#                                 left_index = j - oside_neighborhood
#                                 right_index = j + oside_neighborhood
#                                 ind_portion_adj = portion/(2*oside_neighborhood)
#                                 new_row[j + 1:right_index + 1] += A[i][j + 1:right_index + 1] + A[i][j] * ind_portion_adj
#                                 new_row[left_index:j] += A[i][left_index:j] + A[i][j] * ind_portion_adj
#                                 new_row[j] += A[i][j] * (1-portion)
#                                 index_lst[j][0] = 1
#                                 for sublst in index_lst[j + 1:right_index + 1]:
#                                     sublst[1] = 1
#                                 for sublst in index_lst[left_index:j]:
#                                     sublst[2] = 1
#                     if index_dict[i][j][1] == 0 and index_dict[i][j][2] == 0:
#                         skipped.append(j)
#             for skip in skipped:
#                 if index_lst[skip][1] == 0 and index_lst[skip][2] == 0:
#                     new_row[skip] += A[i][skip]
#             index_dict[i] = index_lst        
#             A[i] = new_row
#         return A, index_dict
        
#     A, index_dict = transfer_heat(A, n_attributes, m, portion, range_percentage)
#     distribute_flag = True
#     while distribute_flag:
#         distribute_flag = False  # Reset the flag
#         for i in range(n_attributes):
#             if any([sublst[0] == 0 for j, sublst in enumerate(index_dict[i]) if 0 < j < m-1]):
#                 distribute_flag = True  # Set the flag if we find any 0
#                 break
#         if distribute_flag:
#             A, index_dict = expand_heat(A, n_attributes, m, portion, range_percentage, index_dict)
#     return A

# def distribute_heat(A, n_attributes, m, portion, range_param):
#     """
#     Distributes the heat in the CA grid.
#     The different counts are the temperature containg a driving force or heat which is transfered according to some defined rules.
#     In other words, it applies some rules that distribute the cells count values.
#     1. Each cell that has a temperature value emits a heat wave in two directions (left & right)
#     2. Heat waves spread to the neighbor cells until they reach the borders of the area determined by the range parameter
#     3. Cells that received energy, emit a portion of the energy they have received, into the direction of the original wave (only)

#     Receive:
#         A: Single CA grid (one-row)
#         n_attributes: Amount of attributes of each data instance
#         m: Amount of cells in CA grid (one-row)
#         portion: Percentage of heat that will be transfered
#         range_param: Percentage of cells that will be utilized in the heat transfer process.
#     Return:
#         A: Updated single CA grid (one-row) with the heat distributed, in other words the counts reaccomodated
#     """
#     for i in range(n_attributes):
#         new_row = np.zeros(m)
#         for j in range(m):
#             if A[i][j] > 0:
#                 left_index = max(0, j - range_param) # Guardian that avoids grabing an out of boundary left_index (beyond 0)
#                 right_index = min(m - 1, j + range_param) 
#                 new_row[left_index:right_index+1] += A[i][j] * portion
#                 new_row[j] += A[i][j] * (1 - portion * (right_index - left_index))
#         A[i] = new_row
#     return A
def distribute_heat(A, n_attributes, m, portion, range_param):
    # range_cells = int(m * range_param)  # Calculate the number of cells based on the percentage
    # for attr in range(n_attributes):
    #     new_row = np.zeros(m)
    #     for cell in range(m):
    #         if A[attr][cell] > 0:
    #             left_index = max(0, cell - range_cells)
    #             right_index = min(m - 1, cell + range_cells)
    #             # Distribute heat to the left and right neighbors
    #             total_cells = right_index - left_index  # Exclude the original cell
    #             heat_per_cell = A[attr][cell] * portion / total_cells
    #             new_row[left_index:right_index + 1] += heat_per_cell
    #             new_row[cell] += A[attr][cell] - heat_per_cell * (total_cells + 1)  # Subtract the distributed heat 
    #     A[attr] = new_row
    # A_list = [arr.tolist() for arr in A]
    return A
    # return A_list

def initialize_CA(n_classes, n_attributes, m):
    """Initialize the Cellular Automata grid."""
    return np.zeros((n_classes, n_attributes, m))
def classify(data_instance, data_min, data_max, CA, n_classes, n_attributes, m):
    """Classify a data point based on the heat values in the CA grid."""
    max_heat = -np.inf
    class_label = -1
    for i in range(n_classes):
        total_heat = 0
        for j in range(n_attributes):
            index = find_cell_index(j, data_instance[j], data_min, data_max, m)
            total_heat += CA[i][j][index]
        if total_heat > max_heat:
            max_heat = total_heat
            class_label = i
    return class_label
def train_CA(X_train, y_train, data_min, data_max, CA, n_classes, n_attributes, m, portion, range_param):
    """Train the Cellular Automata with the training data."""
    for data_instance, label in zip(X_train, y_train):
        CA = map_data_to_CA_cells(data_instance, n_attributes, label, data_min, data_max, CA, m)
    for i in range(n_classes):
        CA[i] = set_temperature_values(CA[i], n_attributes, m)
        CA[i] = distribute_heat(CA[i], n_attributes, m, portion, range_param)
    return CA

class CCA(HyperHeuristic):
    def __init__(self, features: List[str], heuristics: List[str], params: dict):
        super().__init__(features, heuristics)
        self.CA = None
        self.data_min = None
        self.data_max = None
        self.n_classes = params.get('n_classes')
        self.n_attributes = params.get('n_attributes')
        self.m = params.get('m')
        self.portion = params.get('portion')
        self.range_param = params.get('range_param')

    def train(self, filename: str) -> None:
        data = pd.read_csv(filename, header=0)
        columns = ["INSTANCE", "BEST", "ORACLE"] + self._heuristics
        X = data.drop(columns, axis=1).values
        y = data["BEST"].values
        for i in range(len(self._heuristics)):
            y[y == self._heuristics[i]] = i
        y = y.astype("int")
        
        # Normalize the data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        # Get data min and max for CA
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        
        # Train the Cellular Automata classifier
        self.CA = initialize_CA(self.n_classes, self.n_attributes, self.m)
        self.CA = train_CA(X, y, self.data_min, self.data_max, self.CA, self.n_classes, self.n_attributes, self.m, self.portion, self.range_param)

    def getHeuristic(self, problem: Problem) -> str:
        instance = [problem.getFeature(feature) for feature in self._features]
        return self._heuristics[classify(instance, self.data_min, self.data_max, self.CA, self.n_classes, self.n_attributes, self.m)]
    def get_ca_grid(self):
        return self.CA