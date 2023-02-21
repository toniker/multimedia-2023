import wave
import numpy as np
import DCT
import Quantizer
import rle
import huffman
import tonalMasking
import codec0
import sounddevice as sd

import traceback, sys, code

from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from scipy.io.wavfile import write


with wave.open("myfile.wav", "rb") as wave_file:
    # Get number of frames
    num_frames = wave_file.getnframes()
    fs = wave_file.getframerate()
    t_audio = num_frames / fs

    # Read wave file as string of bytes
    wave_data = wave_file.readframes(num_frames)

    # Convert to NumPy array
    wave_data = np.frombuffer(wave_data, dtype=np.int16)

    # Reshape since we have a single channel
    wave_data = wave_data.reshape(num_frames)

    # Load data from file
    h = np.load("h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )

    # Load data from file
    Tq = np.load("Tq.npy", allow_pickle=True)


    M = 32
    N = 36
    L = 512
    number_of_frames = len(wave_data) / (N * M)

    """
    Section 3.1- Sub-band Filtering  
    """
    G = make_mp3_synthesisfb(h, M)
    H = make_mp3_analysisfb(h, M)
    Y_tot, x_hat = codec0.codec0(wave_data, H, M, N)

    # Find reconstructed Y_hat
    Y_hat = np.array([])

    idx = np.arange(0,200).astype('int')
    for i in idx:
        """
        Section 3.2- DCT 
        """
        c = DCT.frameDCT(Y_tot[i*N:(i+1)*N, :])
        """
        Section 3.3- Noise Threshold
        """
        P = DCT.DCTpower(c)
        Kmax = M * N - 1
        Dk = tonalMasking.Dksparse(Kmax)
        Tg = tonalMasking.psycho(c, Dk)

        """
        Section 3.4- Quantization
        """
        cs, sc = Quantizer.DCT_band_scale(c)
        symb_index, SF, B = Quantizer.all_bands_quantizer(c, Tg)

        """
        Section 3.5- Run Length Encoding
        """
        run_symbols = rle.RLEencode(K=len(symb_index), symb_index=symb_index)

        """
        Section 3.6- Huffman Encoding
        """
        frame_stream, frame_symbol_prob = huffman.huff(run_symbols)

        """
        Huffman Decoding
        """
        i_run_symbols = huffman.ihuff(frame_stream, frame_symbol_prob)

        """
        Run Length Decoding
        """
        decoded_symb_index = rle.RLEdecode(K=len(i_run_symbols), run_symbols=i_run_symbols)

        """
        De-quantization
        """
        c_h = Quantizer.all_bands_dequantizer(symb_index, B, SF)
        idct = DCT.iframeDCT(np.expand_dims(c_h, axis=1),M,N)
        if Y_hat.shape[0] == 0:
            Y_hat = idct
        else:
            Y_hat = np.vstack((Y_hat, idct))
        print(i)

    breakpoint()
    final_x_hat = codec0.decoder0(Y_hat,h,M,N)
    sd.play(final_x_hat.astype(np.int16), fs)
    write("final_x_hat.wav", fs, final_x_hat.astype(np.int16))
    breakpoint()
