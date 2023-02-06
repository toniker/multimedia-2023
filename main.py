if __name__ == '__main__':
    import wave
    import numpy as np

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

