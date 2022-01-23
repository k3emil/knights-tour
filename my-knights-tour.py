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
i_moves = [1, 2, 2, 1, -1, -2, -2, -1]
j_moves = [2, 1, -1, -2, -2, -1, 1, 2]


def solve_knights_tour():
    is_solved = False
    options = list()
    i_cur = 0
    j_cur = 0
    counter = 1
    opt = 0

    while not is_solved:
        if counter == 64:
            is_solved = True
            break
        try_next = False
        for it in range(opt, 8):
            i_next = i_cur + i_moves[it]
            j_next = j_cur + j_moves[it]
            if is_valid_position(i_next, j_next):
                counter = counter + 1
                board[i_next][j_next] = counter
                options.append(it)
                i_cur = i_next
                j_cur = j_next
                opt = 0
                try_next = True
                break
        if try_next:
            continue
        else:
            counter = counter - 1
            opt = options.pop()
            board[i_cur][j_cur] = 0
            i_cur = i_cur - i_moves[opt]
            j_cur = j_cur - j_moves[opt]
            opt = opt + 1
    print_solution(is_solved)


def is_valid_position(i_next, j_next):
    if i_next < 0 or i_next > 7 or j_next < 0 or j_next > 7:
        return False
    if str(board[i_next][j_next]) != "0":
        return False
    return True


def print_solution(is_solved):
    for j, row in enumerate(board):
        for i, col in enumerate(row):
            print(str(col) + " ", end="") if len(str(col)) == 2 else print("0" + str(col) + " ", end="")
        print()

    if not is_solved:
        print("\nNOT SOLVED / NO SOLUTION EXISTS")
    else:
        print("\nCONGRATULATIONS ! THE KNIGHTS TOUR PROBLEM is SOLVED !!")


if __name__ == "__main__":
    solve_knights_tour()
