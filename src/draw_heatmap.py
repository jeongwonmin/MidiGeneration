import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


parser = argparse.ArgumentParser()
parser.add_argument('-melody_npy', '-mel', help='your npy file to draw heatmaps')
parser.add_argument('-labels', '-l', default=None, help='chord label npy file')
args = parser.parse_args()

ylabel = [
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
]

chords = [
    "Am", "A#m", "Bm", "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m",
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

def all_chord():
    c = np.eye(12)
    c = np.tile(c, (2,1))
    mm = np.concatenate([np.ones(12), np.zeros(12)])
    mm = mm[:,None]
    chords = np.hstack([c, mm])
    return chords

chord_table = all_chord()

def array_to_chord(chord_array):
    idx = np.where(np.all(chord_table==chord_array, axis=1))[0][0]
    return chords[idx]

def draw_heatmap(melody_array, chord_array, filename):
    fig, ax = plt.subplots(figsize=(4, 6))
    h_min = 48
    h_max = 72
    melody_array = melody_array[h_min:h_max, :]
    heatmap = ax.pcolor(melody_array)
    ax.set_xticks(np.arange(melody_array.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(melody_array.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(np.repeat("", melody_array.shape[1]+1))
    ax.set_yticklabels(ylabel)

    if chord_array is not None:
        ax.set_title(array_to_chord(chord_array))
    plt.savefig(filename)
    plt.close()

def draw_midi(npy_file, label_file):
    npy_data = np.load(npy_file)
    npy_data = np.reshape(npy_data, (npy_data.shape[0], 16, 128))
    npy_data = np.transpose(npy_data, (0, 2, 1))
    save_folder = os.path.splitext(os.path.basename(npy_file))[0]
    save_path = os.path.join(os.path.dirname(npy_file), save_folder+"_heatmap")

    labels = None
    if label_file is not None:
        labels = np.load(label_file)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(npy_data.shape[0]):
        filename = "_".join(["heatmap", str(i)])+".png"
        filename = os.path.join(save_path, filename)
        chord_array = None if labels is None else labels[i]
        draw_heatmap(npy_data[i], chord_array, filename)

if __name__=="__main__":
    draw_midi(args.melody_npy, args.labels)
