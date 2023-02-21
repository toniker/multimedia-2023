import numpy as np
import DCT,codec0
from mp3 import make_mp3_analysisfb
import wave
import tonalMasking
import Quantizer
import matplotlib.pyplot as plt

"""
Section 4- Quantization/Dequantization
Tests the quantization of 3 frames and plots the quantization error
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

    cb = Quantizer.critical_bands(Kmax)

    idx = [2,10,14]

    for i in idx:
        frame = Y_tot[i*N:(i+1)*N, :]
        c = DCT.frameDCT(frame)
        St = tonalMasking.STinit(c,Dk)
        Str,_ =tonalMasking.STreduction(St, c, Tq.reshape(-1, 1))
        Tg = tonalMasking.psycho(c,Dk)
        cs,sc = Quantizer.DCT_band_scale(c)
        symb_index,SF,B = Quantizer.all_bands_quantizer(c,Tg)
        xh = Quantizer.all_bands_dequantizer(symb_index, B, SF)

        # Plot results
        fig1, ax1 = plt.subplots(2, 1, figsize=(15, 15))

        # Plot initial and dequantized DCT components
        fs = 44100

        f_Hz = fs / (2 * M*N) * np.arange(0, Kmax + 1)
        f_barks = tonalMasking.Hz2Barks(f_Hz)

        ax1[0].plot(f_barks,c,label='DCT')
        ax1[0].plot(f_barks,xh,label='DCT_hat')
        # plt.setp(stemlines, 'linestyle', 'dotted')

        ax1[0].set_xlabel(r"Frequency (barks)")
        ax1[0].set_ylabel(r"DCT components")
        ax1[0].set_title(r"Initial and Dequantized DCT components")
        ax1[0].legend()

        # Plot the quantization error
        eb = np.abs(c.T - xh)
        ax1[1].plot(f_barks,eb.reshape(-1,1))
        ax1[1].stem(f_barks[Str],Tg[Str], '-.',linefmt  = 'purple', bottom=-12, label='maskers',basefmt=" ")
        ax1[1].set_xlabel(r"Frequency (barks)")
        ax1[1].set_ylabel(r"|DCT - DCT_hat|")
        ax1[1].set_title(r"Quantization Error of DCT components")
        ax1[1].legend()

        #plt.savefig('media/Quantization' + str(i) + '.pdf')
        plt.show()

    breakpoint()