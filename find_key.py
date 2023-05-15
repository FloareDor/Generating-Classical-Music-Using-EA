import os
import librosa
import numpy as np
import soundfile as sf

def get_keys(songs_dir):
	"""
	Given a directory path containing .wav files, returns a dictionary with the estimated key of each file.
	"""
	keys = {}

	# Iterate over all .wav files in the directory
	for file_name in os.listdir(songs_dir):
		if file_name.endswith(".wav"):
			file_path = os.path.join(songs_dir, file_name)

			# Load the audio file
			y, sr = librosa.load(file_path)

			# Extract the chroma features
			chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

			# Compute the mean of each chroma feature
			chroma_mean = np.mean(chroma, axis=1)

			# Determine the key using music theory rules
			key = np.argmax(chroma_mean)
			mode = "major" if chroma_mean[key] > np.mean(chroma_mean) else "minor"
			freqs = librosa.cqt_frequencies(chroma.shape[0], fmin=librosa.note_to_hz('C1'))
			valid_freqs = freqs > 0.1  # Filter out very small or 0 frequencies
			midi = 12 * (np.log2(np.asanyarray(freqs[valid_freqs])) - np.log2(440.0)) + 69
			key_note = librosa.midi_to_note(midi[key])

			# Add the estimated key to the dictionary
			keys[file_name] = f"{key_note} {mode}"

	return keys

def avg_key(keys):
	key_to_pitch_class = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
						 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
	pitch_class_to_key = {v: k for k, v in key_to_pitch_class.items()}
	# Extract the pitch classes and frequencies from the dictionary
	pitch_classes = []
	frequencies = []
	for key in keys:
		key_note, mode = keys[key].split()
		print(key_note)
		key_note = key_note.replace("1", "")
		if len(key_note) > 1:
			key_note = key_note.replace("♯", "#")
		pitch_class = key_to_pitch_class[key_note]
		pitch_classes.append(pitch_class)
		frequencies.append(librosa.note_to_hz(key_note))

	# Calculate the average pitch class
	avg_pitch_class = round(sum(pitch_classes) / len(pitch_classes))

	# Convert the average pitch class back to its corresponding key
	avg_key_note = pitch_class_to_key[avg_pitch_class]
	
	# Use librosa to estimate the frequency of the average key note
	avg_frequency = librosa.note_to_hz(avg_key_note)
	
	# Use the average frequency to estimate the mode (major or minor) of the average key
	mode = "major" if avg_frequency > 0 else "minor"
	avg_key = f"{avg_key_note} {mode}"

	print(f"The average key is {avg_key}")
	return avg_key

def transpose_files(input_dir, from_key, to_key, output_dir, n_fft=512):
    key_to_pitch_class = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                         'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

    # Determine the pitch difference between the source and target keys
    pitch_diff = key_to_pitch_class[to_key] - key_to_pitch_class[from_key]

    # Iterate over all .wav files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Load the audio file
            y, sr = sf.read(input_path)

            # Transpose the audio to the target key
            y_transposed = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_diff, n_fft=n_fft)

            # Write the transposed audio to a new file
            sf.write(output_path, y_transposed, sr, format='WAV', endian='little', subtype='PCM_16')

if __name__ == "__main__":
	
	songs_dir = "data"
	keys = get_keys(songs_dir)
	avg_key = avg_key(keys)
	print(keys)
	print(avg_key)
	for key in keys:
		print(keys[key])
		transpose_files("data", f"{keys[key].split()[0].replace('1', '').replace('♯', '')}", avg_key.split()[0], "transposed")