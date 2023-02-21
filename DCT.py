import numpy as np
from scipy.fftpack import dct, idct

def frameDCT(Y):
    dct_result = np.array(())
    for row in Y:
        dct_result = np.append(dct_result, dct(row, norm='ortho', overwrite_x=True))
    return np.array(dct_result).reshape(-1, 1)

def iframeDCT(c, M, N):
    idct_result = np.array(())
    for row in c:
        idct_result = np.append(idct_result, idct(row, norm='ortho', overwrite_x=True))
    return np.array(idct_result).reshape(N, M)


def DCTpower(c):
    return 10 * np.log10(c ** 2)
