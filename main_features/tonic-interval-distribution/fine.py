import librosa
import numpy as np

# Load the audio file
audio_file = r'E:\RESEARCH\generate-music\data\1.wav'
y, sr = librosa.load(audio_file)

# Estimate the pitch of the audio signal
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

print(f0)

# Convert the pitch estimates to pitch classes
pitch_classes = librosa.hz_to_midi(f0)
pitch_classes[pitch_classes < 0] = 0
pitch_classes[pitch_classes >= 12] = pitch_classes[pitch_classes >= 12] % 12

# Count the number of occurrences of each pitch class
pitch_class_counts = np.zeros(12)
for pitch_class in pitch_classes:
    pitch_class_counts[int(pitch_class)] += 1

# Calculate the percentage value for each pitch class
percentage_values = pitch_class_counts / np.sum(pitch_class_counts) * 100

# Print the percentage value for each pitch class
for i, pitch_class in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
    print(f'{pitch_class}: {percentage_values[i]:.2f}%')
