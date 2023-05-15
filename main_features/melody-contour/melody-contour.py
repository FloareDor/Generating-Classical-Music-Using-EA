import os
import numpy as np
import music21
import matplotlib.pyplot as plt
import essentia.standard as ess

# Directory where audio files are stored
audio_dir = r'/mnt/e/RESEARCH/generate-music/data'

# Loop through all audio files in directory
try:
	for filename in os.listdir(audio_dir):
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

			print(mean_direction, mean_size, melodic_var, melodic_complexity)
			print(melodic_var, melodic_complexity)
			
			# Plot the mean direction and mean size
			x = plt.figure(filename.replace(".wav", "").replace(".mp3", ""))
			plt.plot(mean_direction, mean_size, 'o')
			plt.title(filename)
			plt.xlabel('Mean Direction')
			plt.ylabel('Mean Size')
			plt.xlim([0.4, 0.6])
			plt.ylim([-0.001, 0.001])
			plt.text(0.29, 0.95, f'Variation: {melodic_var}', transform=plt.gca().transAxes, va='top', ha='center')
			plt.text(0.8, 0.95, f'Complexity: {melodic_complexity}', transform=plt.gca().transAxes, va='top', ha='center')
			plt.text(0.25, 0.35, f'Direction: {mean_direction}', transform=plt.gca().transAxes, va='top', ha='center')
			plt.text(0.8, 0.35, f'Size: {mean_size}', transform=plt.gca().transAxes, va='top', ha='center')
			#plt.text(0.5, 0.95, f'Song: {filename}', transform=plt.gca().transAxes, va='top', ha='center')
			plt.show()
			plt.savefig(r'E:\RESEARCH\generate-music\melody-contour\plots'+filename.replace(".wav", "").replace(".mp3", ""))
except Exception as e:
	print(e)
