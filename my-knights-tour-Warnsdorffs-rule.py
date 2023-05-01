from datetime import datetime

start = datetime.now()
board = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]
i_moves = [-1, 1, 2, 2, 1, -1, -2, -2]
j_moves = [2, 2, 1, -1, -2, -2, -1, 1]


def solve_knights_tour():
    is_solved = False
    options = list()
    i_cur = 4
    j_cur = 4
    counter = 1
    warns_index = 0

    while not is_solved:
        if counter == 64:
            is_solved = True
            break
        for it in range(8):
            i_next = i_cur + i_moves[it]
            j_next = j_cur + j_moves[it]
            if is_valid_position(i_next, j_next):
                for k in range(8):
                    i_next_2 = i_next + i_moves[k]
                    j_next_2 = j_next + j_moves[k]
                    if is_valid_position(i_next_2, j_next_2):
                        warns_index = warns_index + 1
                options.append([it, warns_index])
                warns_index = 0
        min_min = min(options, key=lambda x: x[1])[0]
        i_cur = i_cur + i_moves[min_min]
        j_cur = j_cur + j_moves[min_min]
        counter = counter + 1
        board[i_cur][j_cur] = counter
        options.clear()
    print_solution(is_solved)


def is_valid_position(i_next, j_next):
    if i_next < 0 or i_next > 7 or j_next < 0 or j_next > 7 or str(board[i_next][j_next]) != "0":
        return False
    return True


def print_solution(is_solved):
    for i in range(8):
        for j in range(8):
            print(str(board[i][j]) + " ", end="") if len(str(board[i][j])) == 2 else print("0" + str(board[i][j]) + " ", end="")
        print()

    if not is_solved:
        print("\nNOT SOLVED / NO SOLUTION EXISTS")
    else:
        print("\nCONGRATULATIONS ! THE KNIGHT'S TOUR PROBLEM is SOLVED !!")


if __name__ == "__main__":
    solve_knights_tour()

finish = datetime.now()
print("\nTOTAL EXECUTION TIME = " + str(finish-start))
