import numpy as np


def Dksparse(Kmax):
    Dk = np.zeros((Kmax, Kmax))

    for k in range(Kmax):
        if 2 < k < 282:
            delta_k = [2]
        elif 282 <= k < 570:
            delta_k = [i for i in range(2, 14)]
        elif 570 <= k < 1152:
            delta_k = [i for i in range(2, 28)]
        else:
            delta_k = []

        for j in range(Kmax):
            if j in delta_k:
                Dk[k, j] = 1
    return Dk


def Hz2Barks(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

