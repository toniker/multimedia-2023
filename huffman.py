import numpy as np
import heapq


class Node:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return True

    def children(self):
        return self.left, self.right


def huff(run_symbols):
    symbols, symbol_occurrences = np.unique(run_symbols, axis=0, return_counts=True)

    # Create output arrays
    frame_stream = ""
    frame_symbol_prob = np.zeros((len(symbols), 3))

    # iterate over unique symbols
    for i, symbol in enumerate(symbols):
        frame_symbol_prob[i] = np.hstack((symbol, symbol_occurrences[i]))

    # Organize contents of frame_symbol_prob into a probability and a tuple of the RLE encoded value
    h = [(p, (v, r)) for v, r, p in frame_symbol_prob]
    heapq.heapify(h)

    if len(h) >= 2:
        while len(h) >= 2:
            p1, s1 = heapq.heappop(h)
            p2, s2 = heapq.heappop(h)
            node = Node(s1, s2)
            heapq.heappush(h, (p1 + p2, node))

        # Get the root node of the tree
        _, node = h[0]

        del p1, s1, p2, s2, h, _

    def generate_codes(node, code="", codes={}):
        # If the node is a leaf, save the code for the symbol.
        if isinstance(node, tuple):
            codes[node] = code
            return

        # Traverse the left subtree and append 0 to the code.
        generate_codes(node.left, code + "0", codes)

        # Traverse the right subtree and append 1 to the code.
        generate_codes(node.right, code + "1", codes)

        return codes

    codes = generate_codes(node, '')

    for symbol in run_symbols:
        frame_stream += codes[tuple(symbol)]

    return frame_stream, frame_symbol_prob


def ihuff(frame_stream, frame_symbol_prob):
    run_symbols = np.array([])

    # Organize contents of frame_symbol_prob into a probability and a tuple of the RLE encoded value
    h = [(p, (v, r)) for v, r, p in frame_symbol_prob]
    heapq.heapify(h)

    while len(h) >= 2:
        p1, s1 = heapq.heappop(h)
        p2, s2 = heapq.heappop(h)
        node = Node(s1, s2)
        heapq.heappush(h, (p1 + p2, node))

    # Get the root node of the tree
    _, node = h[0]

    del p1, s1, p2, s2, h, _

    def generate_codes(node, code="", codes={}):
        # If the node is a leaf, save the code for the symbol.
        if isinstance(node, tuple):
            codes[node] = code
            return

        # Traverse the left subtree and append 0 to the code.
        generate_codes(node.left, code + "0", codes)

        # Traverse the right subtree and append 1 to the code.
        generate_codes(node.right, code + "1", codes)

        return codes

    codes = generate_codes(node, '')

    temp_string = ''
    for c in frame_stream:
        temp_string += c
        if temp_string in codes.values():
            for key, value in codes.items():
                if value == temp_string:
                    key = key
                    break
            run_symbols = np.append(run_symbols, np.array(key))
            temp_string = ''

    return run_symbols.reshape((-1, 2))
