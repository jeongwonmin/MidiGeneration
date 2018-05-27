import os
import argparse

import pretty_midi as pm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('npy', help='your npy folder to convert to midi')
args = parser.parse_args()

def save_midi(npy_file):
    npy_data = np.load(npy_file)
    if len(npy_data.shape) != 4:
        return
    npy_data = np.reshape(npy_data, (npy_data.shape[0], 16, 128))
    npy_data = np.transpose(npy_data, (0, 2, 1))
    save_name = os.path.splitext(os.path.basename(npy_file))[0]+".mid"
    save_path = os.path.join(os.path.dirname(npy_file), save_name)

    prgm=pm.Instrument(0)
    mels_arg = np.argmax(npy_data, axis=1)
    for i in range(mels_arg.shape[0]):
        for j in range(mels_arg.shape[1]):
            note=pm.Note(velocity=100, pitch=int(mels_arg[i,j]), start=float((i*mels_arg.shape[1]+j)*0.125), end=float((i*mels_arg.shape[1]+j+1)*0.125))
            prgm.notes.append(note)

    pm2=pm.PrettyMIDI(resolution=96)
    pm2.instruments.append(prgm)
    pm2.write(save_path)

if __name__=="__main__":
    npy_dir = args.npy
    for f in os.listdir(npy_dir):
        if os.path.splitext(f)[-1] == ".npy":
            filename = os.path.join(npy_dir, f)
            save_midi(filename)
