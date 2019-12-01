#!/usr/bin/env python3
import numpy as np
import librosa
import os
import csv

# Create .npy file for X (shape=(N, 128, 128)), and Y (shape=(N,))

X, y = [], []
# Go through metadata, build dict of {filename: class}:
classes = {}
metadata = open("./UrbanSound8K/metadata/UrbanSound8K.csv")
reader = csv.reader(metadata)
for row in reader:
	if row[0] != 'slice_file_name':
		classes[row[0]] = int(row[-2])

# Load .wav files and crop 1d array to 65500 units (so spec is 128X128, ignore too short ones):
target_len = 65500
base_path = "./UrbanSound8K/audio/"
done_count = 0
for fold_dir in os.listdir(base_path):
	for sound in os.listdir(os.path.join(base_path, fold_dir)):
		sound_path = os.path.join(base_path, fold_dir, sound)
		waveform, sr = librosa.load(sound_path)
		if len(waveform) >= target_len:
			split = (len(waveform) - target_len) // 2
			waveform_cropped = waveform[split:split + target_len]
			S = librosa.feature.melspectrogram(y=waveform_cropped, sr=sr, n_mels=128, fmax=8000)
			X.append(S)
			y.append(classes[sound])
			done_count += 1
			print("Done", done_count, "\r", end="")

np.save("X.npy", X)
np.save("y.npy", y)





