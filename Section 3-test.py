import numpy as np
import DCT,codec0
from mp3 import make_mp3_analysisfb
import wave
import tonalMasking

""" 
Section 3- Calculate Hearing Threshold
Plots the hearing threshold for various frames
"""
M = 32
N = 36
L = 512
Tq = np.load("Tq.npy", allow_pickle=True)

with wave.open("myfile.wav", "rb") as wave_file:
    num_frames = wave_file.getnframes()
    wave_data = wave_file.readframes(num_frames)

    # Set the datatype to the sample width
    dtype = np.int16

    # Convert to NumPy array
    wave_data = np.frombuffer(wave_data, dtype=dtype)

    # Reshape since we have a single channel
    wave_data = wave_data.reshape((num_frames, 1))
    h = np.load("h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )
    H = make_mp3_analysisfb(h, M)

    Y_tot, x_hat = codec0.codec0(wave_data, H, M, N)

    Kmax = M*N - 1
    Dk = tonalMasking.Dksparse(Kmax)

    idx = [0,2,6,10,14]
    for i in idx:
        frame = Y_tot[i*N:(i+1)*N, :]
        c = DCT.frameDCT(frame)

        St = tonalMasking.STinit(c, Dk)
        STr, PTr = tonalMasking.STreduction(St, c, Tq.reshape(-1, 1))
        Tg = tonalMasking.psycho(c,Dk)
        tonalMasking.plot_hearing_threshold(Tg,Tq,STr,Kmax,i)
    breakpoint()
