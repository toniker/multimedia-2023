import numpy as np
from scipy.fftpack import dct, idct


def frameDCT(Y):
    return dct(Y, norm='ortho').reshape(-1, 1)


def iframeDCT(c, M, N):
    return idct(c, norm='ortho').reshape(N, M)


def DCTpower(c):
    to_log = lambda x: 10 * np.log10(abs(x)**2)
    return np.vectorize(to_log)(c)
