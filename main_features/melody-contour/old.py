import essentia.standard as ess
import numpy as np
import music21
#/mnt/e/RESEARCH/generate-music/data
# Load the audio signal from a file
loader = ess.MonoLoader(filename='/mnt/e/RESEARCH/generate-music/data/1.wav')
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

# Calculate the mean and standard deviation of the contour
mean_direction = np.mean([d for d, _ in contour_norm])
mean_size = np.mean([s for _, s in contour_norm])
std_direction = np.std([d for d, _ in contour_norm])
std_size = np.std([s for _, s in contour_norm])

# Combine the direction and size features
melodic_contour = [mean_direction, std_direction, mean_size, std_size]

print(melodic_contour)