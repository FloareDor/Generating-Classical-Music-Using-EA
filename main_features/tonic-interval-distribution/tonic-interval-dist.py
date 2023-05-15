import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

data_folder = 'E:\RESEARCH\generate-music\data' 

# define the possible tonic notes
tonic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# define the possible scale degrees
scale_degrees = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']

def calculate_tonic_interval_distribution(og_filename, filename):
	# load the audio file
	y, sr = librosa.load(filename)
	
	# separate harmonic and percussive components
	y_harmonic, y_percussive = librosa.effects.hpss(y)
	
	# calculate the chroma feature from the harmonic component
	C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=12)
	
	# map each pitch class to a MIDI number and count frequency occurrence
	freq_occurrence = {}
	for i, pitch_class in enumerate(scale_degrees):
		midi_note = librosa.note_to_midi(f'{tonic_notes[i % 12]}{int(i / 12) + 2}')
		# map the midi note to the corresponding index in the chroma feature
		chroma_index = midi_note % 12
		freq_occurrence[pitch_class] = np.sum(C[chroma_index])
	
	# normalize frequency occurrence and convert to percentage
	sum_freq_occurrence = sum(freq_occurrence.values())
	normalized_freq_occurrence = {k: v / sum_freq_occurrence * 100 for k, v in freq_occurrence.items()}
	
	# calculate the ratio of sum of percentages of lower case pitch class and the sum of percentages of upper case pitch class
	lower_case_sum = sum([v for k,v in normalized_freq_occurrence.items() if k.islower()])
	upper_case_sum = sum([v for k,v in normalized_freq_occurrence.items() if k.isupper()])
	ratio = lower_case_sum/upper_case_sum
	
	# find mode and top pitch classes
	mode = max(normalized_freq_occurrence, key=normalized_freq_occurrence.get)
	top_pitch_classes = sorted(normalized_freq_occurrence, key=normalized_freq_occurrence.get, reverse=True)[:3]
	
	# print the ratio and return the dictionary
	print(f'Ratio of sum of percentages of lower case pitch class and sum of percentages of upper case pitch class for {og_filename}: {ratio}')
	return normalized_freq_occurrence, mode, top_pitch_classes, og_filename, ratio


# define a function to plot the tonic interval distribution
def plot_tonic_interval_distribution(tonic_interval_distribution, mode, top_pitch_classes, filename, ratio):
	# create a bar plot of the tonic interval distribution
	fig, ax = plt.subplots(figsize=(8, 5))
	colors = ['#FB9999' if c.islower() else '#49BECE' for c in tonic_interval_distribution.keys()]
	ax.bar(tonic_interval_distribution.keys(), tonic_interval_distribution.values(), width=0.6, color=colors, alpha=0.8)
	#ax.bar(tonic_interval_distribution.keys(), tonic_interval_distribution.values(), width=0.6, color='b', alpha=0.8)
	ax.set_xlabel('Scale Degrees')
	ax.set_ylabel('Mean Frequency Occurrence (%)')
	ax.set_title(f'Tonic Interval Distribution of {filename}\nMode: {mode}, Top Pitch Classes: {top_pitch_classes}')
	plt.text(0.5, 0.95, f'm/M Ratio: {ratio}', transform=plt.gca().transAxes, va='top', ha='center')
	print(ratio)
	plt.xticks(rotation=45)
	plt.tight_layout()
	
	# save the plot
	og_filename, ext = os.path.splitext(filename)
	plt.savefig(os.path.join(r'E:\RESEARCH\generate-music\main_features\tonic-interval-distribution\plots', f'{og_filename}_tonic_interval_distribution.png'))

# loop through each audio file in the data folder and calculate its tonic interval distribution
lower_sum = 0
upper_sum = 0
for filename in os.listdir(data_folder):
	if filename.endswith('.mp3') or filename.endswith('.wav'):
		og_filename, ext = os.path.splitext(filename)
		print(f'Calculating tonic interval distribution for {filename}...')
		tonic_interval_distribution, mode, top_pitch_classes, og_filename, ratio = calculate_tonic_interval_distribution(og_filename, os.path.join(data_folder, filename))
		print(f'Tonic interval distribution calculated for {filename}.')
		
		# add the percentage of each pitch class to the appropriate sum
		for pitch_class in tonic_interval_distribution:
			if pitch_class.islower():
				lower_sum += tonic_interval_distribution[pitch_class]
			elif pitch_class.isupper():
				upper_sum += tonic_interval_distribution[pitch_class]
		
		# plot the tonic interval distribution and save the plot
		plot_tonic_interval_distribution(tonic_interval_distribution, mode, top_pitch_classes, filename, ratio)

# calculate the ratio of the sum of percentages of lower case pitch classes and the sum of percentages of upper case pitch classes
ratio = lower_sum / upper_sum
print(f'The ratio of the sum of percentages of lower case pitch classes to the sum of percentages of upper case pitch classes is: {ratio}')
