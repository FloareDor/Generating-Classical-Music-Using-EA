import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

data_folder = 'E:\RESEARCH\generate-music\data' 

# define the possible tonic notes
tonic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# define the possible scale degrees
scale_degrees = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']

tonic_note = 'A'

# define a function to calculate the tonic interval distribution
def calculate_tonic_interval_distribution(og_filename, filename, tonic_note):
    # load the audio file
    y, sr = librosa.load(filename)

    # separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # calculate the chroma feature from the harmonic component
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=12)

    # map each pitch class to a MIDI number and count frequency occurrence
    freq_occurrence = {}
    tonic_index = tonic_notes.index(tonic_note)
    for i, pitch_class in enumerate(scale_degrees):
        midi_note = librosa.note_to_midi(f'{tonic_notes[(tonic_index + i) % 12]}{int((tonic_index + i) / 12) + 2}')
        # map the midi note to the corresponding index in the chroma feature
        chroma_index = midi_note % 12
        freq_occurrence[pitch_class] = np.sum(C[chroma_index])

    # normalize frequency occurrence and convert to percentage
    sum_freq_occurrence = sum(freq_occurrence.values())
    normalized_freq_occurrence = {k: v / sum_freq_occurrence * 100 for k, v in freq_occurrence.items()}

    # find mode and top pitch classes
    mode = max(normalized_freq_occurrence, key=normalized_freq_occurrence.get)
    top_pitch_classes = sorted(normalized_freq_occurrence, key=normalized_freq_occurrence.get, reverse=True)[:3]

    # return the dictionary
    return normalized_freq_occurrence, mode, top_pitch_classes, og_filename



# define a function to plot the tonic interval distribution
def plot_tonic_interval_distribution(tonic_interval_distribution, tonic_note, top_pitch_classes, filename):
    # create a bar plot of the tonic interval distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#49BECE' if c.islower() else '#FB9999' for c in tonic_interval_distribution.keys()]
    ax.bar(tonic_interval_distribution.keys(), tonic_interval_distribution.values(), width=0.6, color=colors, alpha=0.8)
    #ax.bar(tonic_interval_distribution.keys(), tonic_interval_distribution.values(), width=0.6, color='b', alpha=0.8)
    ax.set_xlabel('Scale Degrees')
    ax.set_ylabel('Mean Frequency Occurrence (%)')
    ax.set_title(f'Tonic Interval Distribution of {filename}\nTonic Note: {tonic_note}, Top Pitch Classes: {top_pitch_classes}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # save the plot
    og_filename, ext = os.path.splitext(filename)
    plt.savefig(os.path.join(r'E:\RESEARCH\generate-music\main_features\tonic-interval-distribution\normalized-tonic-note', f'{og_filename}_{tonic_note}.png'))

# define the tonic note to normalize to
# define the desired tonic note

for filename in os.listdir(data_folder):
    if filename.endswith('.mp3') or filename.endswith('.wav'):
        og_filename, ext = os.path.splitext(filename)
        print(f'Calculating tonic interval distribution for {filename}...')
        tonic_interval_distribution, mode, top_pitch_classes, og_filename = calculate_tonic_interval_distribution(og_filename, os.path.join(data_folder, filename), tonic_note)
        print(f'Tonic interval distribution calculated for {filename}.')
        # plot the tonic interval distribution and save the plot
        plot_tonic_interval_distribution(tonic_interval_distribution, mode, top_pitch_classes, filename)



