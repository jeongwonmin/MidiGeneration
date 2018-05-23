import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


parser = argparse.ArgumentParser()
parser.add_argument('npy_file', help='your npy file to draw heatmaps')
args = parser.parse_args()

ylabel = [
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
]    

def draw_heatmap(melody_array, filename):
    fig, ax = plt.subplots(figsize=(4, 6))
    h_min = 48
    h_max = 72
    melody_array = melody_array[h_min:h_max, :]
    heatmap = ax.pcolor(melody_array)
    ax.set_xticks(np.arange(melody_array.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(melody_array.shape[0]) + 0.5, minor=False)

    ax.set_xticklabels(np.repeat("", melody_array.shape[1]+1))
    ax.set_yticklabels(ylabel)
    plt.savefig(filename)
    plt.close()

def draw_midi(npy_file):
    npy_data = np.load(npy_file)
    npy_data = np.reshape(npy_data, (npy_data.shape[0], 16, 128))
    npy_data = np.transpose(npy_data, (0, 2, 1))
    save_folder = os.path.splitext(os.path.basename(npy_file))[0]
    save_path = os.path.join(os.path.dirname(npy_file), save_folder+"_heatmap")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(npy_data.shape[0]):
        filename = "_".join(["heatmap", str(i)])+".png"
        filename = os.path.join(save_path, filename)
        draw_heatmap(npy_data[i], filename)

if __name__=="__main__":
    draw_midi(args.npy_file)
