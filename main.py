if __name__ == '__main__':
    import wave
    import numpy as np
    from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
    import matplotlib.pyplot as plt
    from scipy.signal import CZT
    from scipy import signal

    with wave.open("myfile.wav", "rb") as wave_file:
        # Get number of frames
        num_frames = wave_file.getnframes()

        # Read wave file as string of bytes
        wave_data = wave_file.readframes(num_frames)

        # Set the datatype to the sample width
        dtype = np.int16

        # Convert to NumPy array
        wave_data = np.frombuffer(wave_data, dtype=dtype)

        # Reshape since we have a single channel
        wave_data = wave_data.reshape((num_frames, 1))

        # Load data from file
        h = np.load("h.npy", allow_pickle=True).tolist()['h'].reshape(-1, )

        # Load data from file
        Tq = np.load("Tq.npy", allow_pickle=True)

        # Set number of sub-bands to 32
        M = 32

        """
        Section 3.1- Subband Filtering
        """

        # 1.Calculate analysis and synthesis filters
        G = make_mp3_synthesisfb(h, M)
        H = make_mp3_analysisfb(h, M)

        # 2-3.Compute Frequency Response of Analysis Filters (Hz)
        fs = 44100;
        fig, ax1 = plt.subplots(2,1,figsize=(15,15))

        for i in range(H.shape[1]):
            w, h = signal.freqz(H[:, i])

            # f (Hz) = (w * fs)/2π
            f = w*fs / 2*np.pi
            ax1[0].plot(f, 10 * np.log10(abs(h) ** 2))

            # bark_f = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
            bark_f = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
            ax1[1].plot(bark_f, 10 * np.log10(abs(h) ** 2))
        ax1[0].set_xlabel(r"Frequency (Hz)")
        ax1[0].set_ylabel(r"$10log_{10}(|H(f)|^2 $")
        ax1[0].set_title(r"Frequency Response of Analysis Filters (Hz)")

        ax1[1].set_xlabel(r"Frequency (barks)")
        ax1[1].set_ylabel(r"$10log_{10}(|H(f)|^2 $")
        ax1[1].set_title(r"Frequency Response of Analysis Filters (barks)")

        plt.show()

        breakpoint()
