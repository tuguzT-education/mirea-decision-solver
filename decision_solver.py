"""
Module for decision solving (with realizations of TOPSIS, SAW and ELECTRE I methods)
"""


from typing import List, Tuple, Union, Final
import numpy as np


FloatRow = Union[List[float], Tuple[float, ...], np.ndarray]
BoolRow = Union[List[bool], Tuple[bool, ...], np.ndarray]
Matrix = Union[List[FloatRow], Tuple[FloatRow, ...], np.ndarray]


class DecisionSolver:
    """
    This is a helper class for decision solving methods (SAW, TOPSIS and ELECTRE I)
    """

    def __init__(self, matrix: Matrix, weights: FloatRow, impacts: BoolRow) -> None:
        """
        Basic class constructor

        :param matrix: matrix where rows represent candidates, columns represent criteria
        :param weights: float weights of each criteria
        :param impacts: list of booleans that describes positive or negative impact of each criteria
        """

        self.__rows_num: Final = len(matrix)
        if not self.rows_num:
            raise ValueError('Count of rows must be greater than 0')

        self.__columns_num: Final = len(matrix[0])
        if not self.columns_num:
            raise ValueError('Count of columns must be greater than 0')

        if len(weights) != self.columns_num:
            raise ValueError('Count of weights must be as same as count of criteria '
                             f'({len(weights)} is not equal {self.columns_num})')
        if len(impacts) != self.columns_num:
            raise ValueError('Count of impacts must be as same as count of criteria '
                             f'({len(impacts)} is not equal {self.columns_num})')

        self.__matrix: Final = np.array(matrix, dtype=np.float64)
        self.__impacts: Final = np.array(impacts, dtype=np.bool8)

        self.__weights: Final = np.array(weights, dtype=np.float64)

        # Normalization of weights made by dividing each element on total sum of all elements
        weights_sum = np.sum(self.weights)
        self.__normalized_weights: Final = self.weights / weights_sum if weights_sum else self.weights

    @property
    def rows_num(self):
        return self.__rows_num

    @property
    def columns_num(self):
        return self.__columns_num

    @property
    def matrix(self):
        return self.__matrix

    @property
    def impacts(self):
        return self.__impacts

    @property
    def weights(self):
        return self.__weights

    def __basic_normalized_matrix(self) -> np.ndarray:
        """
        Method for normalizing matrix

        :return: matrix normalized by dividing each of its elements
        on square root from sum of squared elements of appropriate row
        """

        return self.matrix / np.sqrt(np.sum(self.matrix * self.matrix, axis=0))

    def __weighted_normalized_matrix(self, normalized_matrix: Matrix) -> np.ndarray:
        """
        Method for weighting a normalized matrix

        :param normalized_matrix:
        :return: weighted normalized matrix normalized by multiplying
        base normalized matrix on normalized weights
        """

        return np.array(normalized_matrix) * self.__normalized_weights

    def saw(self) -> Tuple[float, ...]:
        """
        SAW solver method

        :return: tuple of candidate indices sorted in descending order
        """

        # Step 1: normalizing matrix to a scale comparable to all existing alternative ratings
        normalized_matrix = np.zeros((self.rows_num, self.columns_num), dtype=np.float64)
        for i in range(self.rows_num):
            row_max, row_min = np.max(self.matrix[i]), np.min(self.matrix[i])
            for j in range(self.columns_num):
                normalized_matrix[i][j] = self.matrix[i][j] / row_max \
                    if self.impacts[j] else row_min / self.matrix[i][j]

        # Step 2: weighting normalized matrix
        weighted_normalized_matrix = self.__weighted_normalized_matrix(normalized_matrix)

        # Step 3: ranking alternatives according to the sum calculated for each row of the matrix
        rank_list = np.sum(weighted_normalized_matrix, axis=1)
        return tuple(np.argsort(rank_list)[::-1])

    def topsis(self) -> Tuple[float, ...]:
        """
        TOPSIS solver method

        :return: tuple of candidate indices sorted in descending order
        """

        # Step 1: weighting normalized matrix
        normalized_matrix = self.__basic_normalized_matrix()
        weighted_normalized_matrix = self.__weighted_normalized_matrix(normalized_matrix)

        # Step 2: calculating the best and the worst alternatives
        worst_alternative = np.zeros(self.columns_num, dtype=np.float64)
        best_alternative = np.zeros(self.columns_num, dtype=np.float64)
        for j in range(self.columns_num):
            sliced = weighted_normalized_matrix[:, j]
            if self.impacts[j]:
                worst_alternative[j], best_alternative[j] = np.min(sliced), np.max(sliced)
            else:
                worst_alternative[j], best_alternative[j] = np.max(sliced), np.min(sliced)

        # Step 3.1: calculating distance for each elements between matrix and the worst alternative
        distance_worst = np.sqrt(np.sum(np.square(weighted_normalized_matrix - worst_alternative), axis=1))
        # Step 3.2: calculating distance for each elements between matrix and the best alternative
        distance_best = np.sqrt(np.sum(np.square(weighted_normalized_matrix - best_alternative), axis=1))

        # Step 4: calculating similarity to the worst condition
        similarity_worst = distance_worst / (distance_worst + distance_best)
        # Step 5: ranking alternatives according to the similarity
        return tuple(np.argsort(similarity_worst)[::-1])

    def electre_1(self) -> Tuple[float, ...]:
        """
        ELECTRE I solver method

        :return: tuple of candidate indices sorted in descending order
        """

        # Step 1: weighting normalized matrix
        normalized_matrix = self.__basic_normalized_matrix()
        weighted_normalized_matrix = self.__weighted_normalized_matrix(normalized_matrix)

        # Step 2: calculating the concordance matrix
        concordance = np.zeros((self.rows_num, self.rows_num), dtype=np.float64)
        for i in range(self.rows_num):
            for j in range(self.rows_num):
                if i != j:
                    for k in range(self.columns_num):
                        condition = (self.impacts[k] and
                                     weighted_normalized_matrix[i][k] > weighted_normalized_matrix[j][k]) or \
                                    (not self.impacts[k] and
                                     weighted_normalized_matrix[i][k] < weighted_normalized_matrix[j][k])
                        if condition:
                            concordance[i, j] += self.__normalized_weights[k]
                else:
                    concordance[i, j] = 1
        # Step 3: calculating the discordance matrix
        discordance = np.zeros((self.rows_num, self.rows_num), dtype=np.float64)
        for i in range(self.rows_num):
            for j in range(self.rows_num):
                if i != j:
                    for k in range(self.columns_num):
                        condition = (self.impacts[k] and
                                     weighted_normalized_matrix[i][k] <= weighted_normalized_matrix[j][k]) or \
                                    (not self.impacts[k] and
                                     weighted_normalized_matrix[i][k] >= weighted_normalized_matrix[j][k])
                        if condition:
                            discordance[i][j] = np.max(weighted_normalized_matrix[j] - weighted_normalized_matrix[i])

        min_concordance, max_discordance = 1.0, 0.0
        count = 10
        # Rank list of candidates score defined by
        # belonging to the set of core binary relations
        rank_list = [count for _ in range(self.rows_num)]
        for k in range(count):
            # Step 4.1: calculating the binary dominance matrix using
            # minimum value of concordance and maximum value of discordance
            dominance = np.zeros((self.rows_num, self.rows_num), dtype=np.bool8)
            for i in range(self.rows_num):
                for j in range(self.rows_num):
                    if (i != j) and (concordance[i][j] >= min_concordance) and (discordance[i][j] <= max_discordance):
                        dominance[i][j] = True
            # Step 4.2: assigning a score to alternative if row
            # has a dominance by any criterion and it has no score
            for i in range(self.rows_num):
                if True in dominance[i, :] and rank_list[i] == count:
                    rank_list[i] = k
            tmp = 1 / count
            # Step 4.3: change minimum value of concordance and maximum value of discordance
            # to enlarge core of binary relations on the next step
            min_concordance -= tmp
            max_discordance += tmp

        # Step 5: rank alternatives according to their score
        # (the less the score, the better is alternative)
        return tuple(np.argsort(rank_list))
