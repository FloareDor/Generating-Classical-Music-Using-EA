import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

parent_folder = 'E:/RESEARCH/generate-music/data'

# define the possible tonic notes
tonic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# define the possible scale degrees
scale_degrees = ['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N']

label_map = {'KARUNA': '#FF5733' , 'SHANTHA': '#00C5CD', 'SHRINGAR': '#F1C40F', 'VEERA': '#8E44AD'}
y_values = {}
for ind, rasa in enumerate(label_map.keys()):
    y_values[rasa] = ind

def calculate_mean_energy_distribution(og_filename, filename):
	# load the audio file
	y, sr = librosa.load(filename)
	
	# calculate the chromagram
	chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
	
	# calculate the mean value of each pitch class
	mean_values = np.mean(chroma, axis=1)
	
	normalized_mean_values = mean_values / np.sum(mean_values)
	# create a dictionary mapping scale degrees to mean values
	interval_distribution = dict(zip(scale_degrees, normalized_mean_values * 100))
	
	# calculate sum of energies for upper case letters and lower case letters
	upper_energy_sum = sum([interval_distribution[c] for c in interval_distribution if c.isupper()])
	lower_energy_sum = sum([interval_distribution[c] for c in interval_distribution if c.islower()])
	
	# calculate the ratio of upper to lower energy sums
	energy_ratio = upper_energy_sum / lower_energy_sum
	
	mode = max(interval_distribution, key=interval_distribution.get)
	top_notes = sorted(interval_distribution, key=interval_distribution.get, reverse=True)[:3]
	
	# return the dictionary and energy ratio
	return interval_distribution, mode, top_notes, og_filename, energy_ratio


def plot_mean_energy_distribution(interval_distribution, mode, top_notes, filename, ind, energy_ratio):
    colors = ['#49BECE' if key.isupper() else '#FB9999' for key in interval_distribution]

    plt.bar(interval_distribution.keys(), interval_distribution.values(), color=colors)

    plt.text(0.3, 0.95, f'Mode: {mode}', transform=plt.gca().transAxes, va='top', ha='center')
    plt.text(0.7, 0.95, f'Top Notes: {top_notes}', transform=plt.gca().transAxes, va='top', ha='center')
    plt.text(0.5, 0.995, f'Song: {ind}: {filename.replace(".png","")[-5:]}', transform=plt.gca().transAxes, va='top', ha='center')
    
    # add energy ratio to plot
    plt.text(0.5, 0.05, f'Energy Ratio(M/m): {energy_ratio:.2f}', transform=plt.gca().transAxes, va='bottom', ha='center')

    plt.xlabel('Scale degree')
    plt.ylabel('Mean energy distribution (%)')
    
    # return the figure object
    return plt.gcf()



# loop over all files in the directory
energy_ratios = []
rasa_labels = []
for rasa_folder in ['KARUNA', 'SHANTHA', 'SHRINGAR', 'VEERA']:
    # define the input and output folders for the current rasa
    input_folder = os.path.join(parent_folder, rasa_folder)
    output_folder = os.path.join(parent_folder, rasa_folder+'_output')
    os.makedirs(output_folder, exist_ok=True)

    ind = 0
    # loop over all files in the current rasa folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            print(ind)
            # get the full path to the file
            filepath = os.path.join(input_folder, filename)

            # calculate the mean energy distribution
            interval_distribution, mode, top_notes, og_filename, energy_ratio = calculate_mean_energy_distribution(filename, filepath)
            energy_ratios.append(energy_ratio)
            rasa_labels.append(rasa_folder)

            # plot the mean energy distribution and save the plot
            output_filename = os.path.join(output_folder, filename.replace(".wav", ".png").replace(".mp3", ".png"))
            fig = plot_mean_energy_distribution(interval_distribution, mode, top_notes, output_filename, ind, energy_ratio=energy_ratio)
            #fig.savefig(output_filename, dpi=300)
            plt.clf()  # clear the figure after saving it
            ind += 1
            
energy_ratios = np.array(energy_ratios)
rasa_labels = np.array(rasa_labels)

color_labels = []
for rasa_label in rasa_labels:
    color_labels.append(label_map[rasa_label])
    
color_labels = np.array(color_labels)
unique_rasa_labels = np.unique(rasa_labels)

# create an array of indices for each unique label
yticks_indices = range(len(unique_rasa_labels))

# set the yticks to the indices and the labels to the unique rasa labels
plt.yticks(yticks_indices, unique_rasa_labels)

# plot a scatter plot of the energy ratios, colored by rasa label
plt.scatter(energy_ratios, [yticks_indices[np.where(unique_rasa_labels == label)[0][0]] for label in rasa_labels], c=color_labels)

# add a title and labels to the plot
plt.title('Energy ratio distribution by rasa')
plt.xlabel('Energy ratio (M/m)')
plt.ylabel('Rasa')

# show the plot
plt.show()