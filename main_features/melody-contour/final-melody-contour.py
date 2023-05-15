import os
import numpy as np
import music21
import matplotlib.pyplot as plt
import essentia.standard as ess

# Define a dictionary to map rasa to color
rasa_color = {
	'KARUNA': 'red',
	'SHANTHA': 'blue',
	'SHRINGAR': 'green',
	'VEERA': 'purple'
}
variation_dict = {'KARUNA': [], 'SHANTHA': [], 'SHRINGAR': [], 'VEERA': []}
complexity_dict = variation_dict
size_dict = complexity_dict
dir_dict = size_dict
# Lists to store feature values for all audio files
mean_directions = []
mean_sizes = []
melodic_vars = []
melodic_complexities = []
audio_dirs = []
# Loop through all folders containing audio files
for rasa in rasa_color:
	print(rasa)
	# Directory where audio files are stored
	audio_dir = f"/mnt/e/RESEARCH/generate-music/data/{rasa}"
	print(os.path.basename(os.path.dirname(audio_dir)))

	# Loop through all audio files in directory
	try:
		for filename in os.listdir(audio_dir):
			print(os.path.basename(os.path.dirname(audio_dir)))
			
			if filename.endswith('.mp3') or filename.endswith('.wav'):
				print(filename)
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

				variation_dict[rasa].append(melodic_var)
				complexity_dict[rasa].append(melodic_complexity)
				size_dict[rasa].append(mean_size)
				dir_dict[rasa].append(mean_direction)

	except Exception as e:
		print(e)

final_json = {}

final_json["variation"] = variation_dict
final_json["complexity"] = complexity_dict
final_json["mean_size"] = size_dict
final_json["mean_direction"] = dir_dict

with open("final.json", "w") as outfile:
    outfile.write(final_json)

# create a dictionary to store the feature values for each rasa

# create a list of colors corresponding to each rasa
colors = [rasa_color[rasa] for rasa in variation_dict.keys()]

# create the scatter plot
fig, ax = plt.subplots()
for rasa in variation_dict.keys():
	x = variation_dict[rasa]
	y = [rasa] * len(x)
	ax.scatter(x, y, c=rasa_color[rasa], label=rasa)
ax.set_xlabel('Melodic Variation')
ax.set_ylabel('Rasa')
ax.set_title('Melodic Variation by Rasa')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for rasa in variation_dict.keys():
	x = complexity_dict[rasa]
	y = [rasa] * len(x)
	ax.scatter(x, y, c=rasa_color[rasa], label=rasa)
ax.set_xlabel('Melodic Complexity')
ax.set_ylabel('Rasa')
ax.set_title('Melodic Complexity by Rasa')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for rasa in variation_dict.keys():
	x = size_dict[rasa]
	y = [rasa] * len(x)
	ax.scatter(x, y, c=rasa_color[rasa], label=rasa)
ax.set_xlabel('Mean Size')
ax.set_ylabel('Rasa')
ax.set_title('Mean Size by Rasa')
ax.legend()
plt.show()

fig, ax = plt.subplots()
for rasa in variation_dict.keys():
	x = dir_dict[rasa]
	y = [rasa] * len(x)
	ax.scatter(x, y, c=rasa_color[rasa], label=rasa)
ax.set_xlabel('Mean Direction')
ax.set_ylabel('Rasa')
ax.set_title('Mean Direction by Rasa')
ax.legend()
plt.show()
