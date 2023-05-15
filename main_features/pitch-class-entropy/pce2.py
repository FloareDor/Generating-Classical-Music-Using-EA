import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

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

# Set directories of audio files for each rasa
audio_dirs = {
    "KARUNA": "E:\RESEARCH\generate-music\data\KARUNA",
    "SHANTHA": "E:\RESEARCH\generate-music\data\SHANTHA",
    "SHRINGAR": "E:\RESEARCH\generate-music\data\SHRINGAR",
    "VEERA": "E:\RESEARCH\generate-music\data\VEERA"
}

# Iterate through all audio files and extract the pce and rasa for each file
data = []
for rasa, audio_dir in audio_dirs.items():
    for filename in os.listdir(audio_dir):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            filepath = os.path.join(audio_dir, filename)
            pce = calculate_pce(filepath)
            data.append((rasa, pce))

# Create scatter plot of pitch class entropy values and color code the points based on the rasa
rasas = ["KARUNA", "SHANTHA", "SHRINGAR", "VEERA"]
colors = ["red", "blue", "green", "orange"]
for i, rasa in enumerate(rasas):
    rasa_data = [d for d in data if d[0] == rasa]
    x = [d[1] for d in rasa_data]
    y = [i] * len(rasa_data)
    plt.scatter(x, y, color=colors[i], label=rasa)

plt.xlabel("Pitch Class Entropy")
plt.ylabel("Rasa")
plt.yticks(range(len(rasas)), rasas)
plt.legend()
plt.show()
