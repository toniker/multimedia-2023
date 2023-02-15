import numpy as np
import DCT,codec0
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
import wave
import tonalMasking

M = 32
N = 36
L = 512

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
    breakpoint()

    Kmax = M*N - 1
    Dk = tonalMasking.Dksparse(Kmax)
    frame = Y_tot[4*N:5*N, :]
    c = DCT.frameDCT(frame)
    p = tonalMasking.DCTpower(c)
    St = tonalMasking.STinit(c,Dk)
    PM = tonalMasking.MaskPower(c,St)
    breakpoint()
