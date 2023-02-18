import numpy as np
import nothing
from frame import frame_sub_analysis, frame_sub_synthesis


def codec0(wavin, h, M, N):
    Y_tot = coder0(wavin, h, M, N)
    x_xat = decoder0(Y_tot, h, M, N)

    return Y_tot, x_xat


def coder0(wavin, h, M, N):
    L, M = h.shape

    wave_buffer = np.zeros(M * N + L)
    number_of_frames = len(wavin) / (N * M)
    Y_tot = np.array([])

    for i in range(int(number_of_frames)):
        wave_buffer = np.roll(wave_buffer, -N * M)
        wave_buffer[-N * M:] = wavin[
                               i * N * M:(i + 1) * N * M].flatten()  # Is this (i * N * M:(i + 1) * N * M) correct?
        y = frame_sub_analysis(wave_buffer, h, N)
        Yc = nothing.donothing(y)

        if Y_tot.shape[0] == 0:
            Y_tot = Yc
        else:
            Y_tot = np.vstack((Y_tot, Yc))

    return Y_tot


def decoder0(Y_tot, h, M, N):
    L, M = h.shape

    Y_buffer = np.zeros([N + int(L / M), M])
    number_of_frames = Y_tot.shape[0] // N
    x_hat = np.array([])

    Yh = nothing.idonothing(Y_tot)  # This is not done for every frame

    for i in range(int(number_of_frames)):
        Y_buffer = np.roll(Y_buffer, -N, axis=0)
        Y_buffer[-N:, :] = Yh[i * N:(i + 1) * N, :]
        x_hat = np.append(x_hat, frame_sub_synthesis(Y_buffer, h))

    return x_hat
