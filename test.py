import numpy as np
import DCT,codec0
import huffman
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
import wave
import tonalMasking
import Quantizer
import rle

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

    num_of_frames = len(wave_data) / (N * M)
    Kmax = M*N - 1
    Dk = tonalMasking.Dksparse(Kmax)


    frame = Y_tot[4*N:5*N, :]
    c = DCT.frameDCT(frame)
    Tg = tonalMasking.psycho(c,Dk)
    cs,sc = Quantizer.DCT_band_scale(c)
    # s = Quantizer.quantizer(cs,4)
    # d = Quantizer.dequantizer(s,4)
    symb_index,SF,B = Quantizer.all_bands_quantizer(c,Tg)
    xh = Quantizer.all_bands_dequantizer(symb_index, B, SF)

    eb = np.abs(c.T - xh)
    P_eb = 10 * np.log10(eb ** 2)
    MSE = np.mean(eb)

    symb_index = np.array([0, 0, 1, 2, 3, 3, 2, 0, 0, 0, 0, 2, 3, 4])
    run_symbols = rle.RLEencode(K=len(symb_index), symb_index=symb_index)
    decoded_symb_index = rle.RLEdecode(K=len(run_symbols), run_symbols=run_symbols)

    frame_stream, frame_symbol_prob = huffman.huff(run_symbols)
    run_symbols2 = huffman.ihuff(frame_stream, frame_symbol_prob)
    breakpoint()
