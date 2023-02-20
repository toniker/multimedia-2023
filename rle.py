import numpy as np


def RLEencode(symb_index, K):
    run_symbols = np.zeros((K, 2))
    i = 0

    while i <= len(symb_index) - 1:
        count = 1
        char = symb_index[i]
        j = i
        while j < len(symb_index) - 1:
            if symb_index[j] == symb_index[j + 1]:
                count = count + 1
                j = j + 1
            else:
                break

        run_symbols[i, 0] = char
        run_symbols[i, 1] = count
        i = j + 1

    run_symbols = run_symbols[~np.all(run_symbols == 0, axis=1)]
    return run_symbols


def RLEdecode(run_symbols, K):
    symb_index = np.array([])

    for i in range(len(run_symbols)):
        symb_index = np.append(symb_index, np.repeat(run_symbols[i, 0], run_symbols[i, 1]))

    return symb_index.astype('int64')
