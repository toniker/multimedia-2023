import numpy as np
import heapq


class Node:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right


def huff(run_symbols):
    # Get all unique symbols and their number of occurrences
    symbols = np.unique(run_symbols[:, 0])
    symbol_occurrences = np.sum(run_symbols[:, 1])

    # Create output arrays
    frame_stream = np.array([])
    frame_symbol_prob = np.zeros((len(symbols), 3))

    # iterate over unique symbols
    for i, symbol in enumerate(symbols):
        # Count all occurrences of this symbol
        count = np.sum(run_symbols[run_symbols[:, 0] == symbol, 1])

        frame_symbol_prob[i, 0] = symbol
        frame_symbol_prob[i, 1] = count
        frame_symbol_prob[i, 2] = count / symbol_occurrences

    h = [(p, s) for s, _, p in frame_symbol_prob]
    heapq.heapify(h)

    while len(h) >= 2:
        p1, s1 = heapq.heappop(h)
        p2, s2 = heapq.heappop(h)
        node = Node(s1, s2)
        heapq.heappush(h, (p1 + p2, node))

    return frame_symbol_prob
