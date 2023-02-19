import numpy as np


def huff(run_symbols):
    symbols = np.unique(run_symbols[:, 0])
    symbol_occurrences = np.sum(run_symbols[:, 1])

    frame_stream = np.array([])
    frame_symbol_prob = np.zeros((len(symbols), 3))

    # iterate over unique symbols and sum their occurrences
    for i, symbol in enumerate(symbols):
        count = np.sum(run_symbols[run_symbols[:, 0] == symbol, 1])
        frame_symbol_prob[i, 0] = symbol
        frame_symbol_prob[i, 1] = count
        frame_symbol_prob[i, 2] = count / symbol_occurrences

    return frame_symbol_prob
