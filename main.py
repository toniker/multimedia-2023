if __name__ == '__main__':
    import wave
    import numpy as np
    from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb

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

        H = make_mp3_analysisfb(h, M)
        G = make_mp3_synthesisfb(h, M)

        breakpoint()

