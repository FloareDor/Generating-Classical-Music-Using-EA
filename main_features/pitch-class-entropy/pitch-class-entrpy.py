import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
from statistics import mean
# Function to calculate pitch class entropy of an audio file
def calculate_pce(filepath, sr=22050, hop_length=512):
    # Load audio file
    y, sr = librosa.load(filepath, sr=sr)

    # Calculate pitch chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Calculate pitch class histogram
    pitch_class_histogram = np.sum(chroma, axis=1)

    # Calculate pitch class probabilities
    total_notes = np.sum(pitch_class_histogram)
    pitch_class_probabilities = pitch_class_histogram / total_notes

    # Calculate pitch class entropy
    pitch_class_entropy = -np.sum(pitch_class_probabilities * np.log2(pitch_class_probabilities))

    return pitch_class_entropy

# Set directory of audio files
audio_dir = "E:\RESEARCH\generate-music\data\VEERA"

# Calculate pitch class entropy of all audio files in directory
pces = []
for i, filename in enumerate(os.listdir(audio_dir), 1):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        filepath = os.path.join(audio_dir, filename)
        pce = calculate_pce(filepath)
        pces.append(pce)
print(mean(pces))
# Plot pitch class entropy values
plt.plot(range(1, len(pces) + 1), pces)
plt.xlabel("File")
plt.ylabel("Pitch Class Entropy")
plt.title("Pitch Class Entropy of Audio Files")
plt.show()
