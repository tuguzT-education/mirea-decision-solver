import decision_solver

matrix = (
    (8, 8, 7, 6, 5, 9),
    (7, 10, 9, 10, 9, 8),
    (6, 6, 9, 10, 10, 8),
    (5, 10, 8, 10, 9, 7),
    (5, 6, 6, 10, 10, 6),
    (6, 8, 8, 10, 8, 7),
)
weights = (3, 4, 5, 3, 5, 4,)
impacts = (True, True, True, True, True, True,)

if __name__ == '__main__':
    solver = decision_solver.DecisionSolver(matrix, weights, impacts)
    print('TOPSIS solution:   ', solver.topsis())
    print('ELECTRE I solution:', solver.electre_1())
    print('SAW solution:      ', solver.saw())
