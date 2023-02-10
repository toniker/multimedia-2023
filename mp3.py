import numpy as np

import nothing
from frame import frame_sub_analysis, frame_sub_synthesis


def make_mp3_analysisfb(h: np.ndarray, M: int) -> np.ndarray:
    """
    @:param h η κρουστική απόκριση του πρότυπου βαθυπερατού φίλτρου
    @:param M ο αριθμός των ζωνών στις οποίες θα γίνει ο διαχωρισμός.
    @:return  πίνακας διάστασης L × M, όπου L το μήκος της κρουστικής απόκρισης του φίλτρου,
    ο οποίος ως στήλες έχει τις κρουστικές αποκρίσεις των φίλτρων hi(n), i=1,...,M
    """

    H = np.zeros([len(h), M], dtype=np.float32)

    for i in range(1, M + 1):
        n = np.arange(h.shape[0], dtype=np.int64)
        freq_i = (2 * i - 1) * np.pi / (2.0 * M)
        phas_i = -(2 * i - 1) * np.pi / 4.0
        tmp = np.cos(freq_i * n + phas_i)
        x = np.multiply(h, tmp)
        H[:, i - 1] = x
    return H


def make_mp3_synthesisfb(h: np.ndarray, M: int) -> np.ndarray:
    H = make_mp3_analysisfb(h, M)
    L = len(h)
    G = np.flip(H, axis=0)
    return G


def coder0(wavin, h, M, N):
    L, M = h.shape

    wave_buffer = np.zeros(M * N + L)
    number_of_frames = len(wavin) / (N * M)
    Y_tot = np.array([])

    for i in range(int(number_of_frames)):
        wave_buffer = np.roll(wave_buffer, -N * M)
        wave_buffer[-N * M:] = wavin[i * N * M:(i + 1) * N * M].flatten()
        y = frame_sub_analysis(wave_buffer, h, N)
        Yc = nothing.donothing(y)

        if Y_tot.shape[0] == 0:
            Y_tot = Yc
        else:
            Y_tot = np.vstack((Y_tot, Yc))

    return Y_tot

def decoder0(Y_tot, h, M, N):
    L, M = h.shape

    Y_buffer = np.zeros([N + int(L/M), M])
    number_of_frames = Y_tot.shape[0] // N
    x_hat = np.array([])

    Yh = nothing.idonothing(Y_tot)

    for i in range(int(number_of_frames)):
        Y_buffer = np.roll(Y_buffer, -N, axis=0)
        Y_buffer[-N:,:] = Yh[i * N:(i + 1) * N,:]
        x_hat = np.append(x_hat,frame_sub_synthesis(Y_buffer, h))

    return x_hat


