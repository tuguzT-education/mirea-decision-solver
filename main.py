import decision_solver

matrix = (
    (8, 8, 7, 6, 5, 9),    # Unity
    (7, 10, 9, 10, 9, 8),  # Unreal Engine 4
    (6, 6, 9, 10, 10, 8),  # Godot
    (5, 10, 8, 10, 9, 7),  # CryEngine 5
    (5, 6, 6, 10, 10, 6),  # Blender Game Engine
    (6, 8, 8, 10, 8, 7),   # Source 2
)
weights = (
    3,  # Простота освоения
    4,  # Функциональность
    5,  # Надежность
    3,  # Расширяемость
    5,  # Приобретение лицензии
    4,  # Удобство использования
)
impacts = (True, True, True, True, True, True,)

if __name__ == '__main__':
    solver = decision_solver.DecisionSolver(matrix, weights, impacts)
    print('TOPSIS solution:   ', solver.topsis())
    print('ELECTRE I solution:', solver.electre_1())
    print('SAW solution:      ', solver.saw())
