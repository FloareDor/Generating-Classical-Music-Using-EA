import os
import numpy as np
import music21
import matplotlib.pyplot as plt
import essentia.standard as ess

# Directory where audio files are stored
audio_dir = r'/mnt/e/RESEARCH/generate-music/data/KARUNA'

# Lists to store feature values for all audio files
mean_directions = []
mean_sizes = []
melodic_vars = []
melodic_complexities = []

# Loop through all audio files in directory
try:
	for filename in os.listdir(audio_dir):
		print(filename)
		if filename.endswith('.mp3') or filename.endswith('.wav'):
			
			# Load the audio signal from file
			loader = ess.MonoLoader(filename=os.path.join(audio_dir, filename))
			audio = loader()
			
			# Extract the pitches from the audio signal using the Melodia algorithm
			pitch_extractor = ess.PredominantPitchMelodia()
			pitch_values, _ = pitch_extractor(audio)

			# Convert the pitches to a list of note objects
			notes = []
			for pitch in pitch_values:
				if pitch > 0:
					note = music21.pitch.Pitch()
					note.frequency = pitch
					notes.append(note)

			# Calculate the direction and size of pitch movements
			contour = []
			for i in range(1, len(notes)):
				interval = music21.interval.Interval(noteStart=notes[i-1], noteEnd=notes[i])
				direction = interval.direction
				size = interval.semitones
				contour.append((direction, size))

			# Normalize the contour
			contour_norm = [(d/2 + 0.5, s/12) for d, s in contour]

			# Calculate the mean direction and mean size of the contour
			mean_direction = np.mean([d for d, _ in contour_norm])
			mean_size = np.mean([s for _, s in contour_norm])
			std_direction = np.std([d for d, _ in contour_norm])
			std_size = np.std([s for _, s in contour_norm])
			
			# Calculate the melodic variation
			melodic_var = np.var([s for _, s in contour_norm])

			# Calculate the melodic complexity
			intervals = [interval for _, interval in contour]
			melodic_complexity = len(set(intervals))

			# Append feature values to lists
			mean_directions.append(mean_direction)
			mean_sizes.append(mean_size)
			melodic_vars.append(melodic_var)
			melodic_complexities.append(melodic_complexity)
			
	# Plot variation feature values for all audio files
	fig, ax = plt.subplots()
	ax.bar(range(len(melodic_vars)), melodic_vars)
	ax.set_xticks(range(len(melodic_vars)))
	ax.set_xticklabels(os.listdir(audio_dir))
	ax.set_xlabel('Audio Files')
	ax.set_ylabel('Melodic Variation')
	ax.set_title('Melodic Variation for All Audio Files')
	plt.savefig(audio_dir + r'melodic_variation.png')
	plt.show()

	# Plot complexity feature values for all audio files
	fig, ax = plt.subplots()
	ax.bar(range(len(melodic_complexities)), melodic_complexities)
	ax.set_xticks(range(len(melodic_complexities)))
	ax.set_xticklabels(os.listdir(audio_dir))
	ax.set_xlabel('Audio Files')
	ax.set_ylabel('Melodic Complexity')
	ax.set_title('Melodic Complexity for All Audio Files')
	plt.savefig(audio_dir + r'melodic_complexity.png')
	plt.show()

	# Plot size feature values
	fig, ax = plt.subplots()
	ax.bar(range(len(mean_sizes)), mean_sizes)
	ax.set_xticks(range(len(mean_sizes)))
	ax.set_xticklabels(os.listdir(audio_dir))
	ax.set_xlabel('Audio Files')
	ax.set_ylabel('Mean Size')
	ax.set_title('Mean Size for All Audio Files')
	plt.savefig(audio_dir + r'mean_size.png')
	plt.show()

	fig, ax = plt.subplots()
	ax.bar(range(len(mean_directions)), mean_directions)
	ax.set_xticks(range(len(mean_directions)))
	ax.set_xticklabels(os.listdir(audio_dir))
	ax.set_xlabel('Audio Files')
	ax.set_ylabel('Mean Direction')
	ax.set_title('Mean Direction for All Audio Files')
	plt.savefig(audio_dir + r'mean_direction.png')
	plt.show()

except Exception as e:
	print(e)
