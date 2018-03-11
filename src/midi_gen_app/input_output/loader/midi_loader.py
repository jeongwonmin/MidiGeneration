import os

import pretty_midi as pm
import numpy as np
import pandas as pd


class MidiLoader(object):
    def __init__(self, midi_path, genres=["pops"]):
        self._midi_data = {}
        for g in genres:
            self._midi_data.update({g:{}})
            folder = os.path.join(midi_path, g)
            paths = []
            for mid_f in os.listdir(folder):
                if os.path.splitext(mid_f)[1].find("mid") != -1:
                    paths.append(os.path.join(folder, mid_f))
            self._midi_data[g].update({"files":paths})

        bpm_folder = os.path.join(midi_path, "bpms")
        for g in genres:
            bpm_f = g+".csv"
            bpm_f = os.path.join(bpm_folder, bpm_f)
            df = pd.read_csv(bpm_f, header=None)
            df.set_index(0, inplace=True)
            self._midi_data[g].update({"bpms":df})

        self._add_piano_roll()

    def _add_piano_roll(self):
        def get_fs(bpm):
            return 1/ (60/(bpm * 4))

        def get_piano_roll(filepath, bpm):
            midi_obj = pm.PrettyMIDI(midi_file=filepath)
            insts = midi_obj.instruments
            melody_obj = insts[0]
            chord_obj = insts[1]
            melody = melody_obj.get_piano_roll(fs=get_fs(bpm))
            chord = chord_obj.get_piano_roll(fs=get_fs(bpm))
            return melody, chord

        for k, i in self._midi_data.items():
            self._midi_data[k].update({"piano_roll":{}})
            mids = self._midi_data[k]["files"]
            bpms = self._midi_data[k]["bpms"]
            for m in mids:
                basename = os.path.basename(m)
                bpm = bpms.ix[basename].values[0]
                melody, chord = get_piano_roll(m, bpm)
                self._midi_data[k]["piano_roll"].update({
                    basename:{
                        "melody": melody,
                        "chord": chord,
                      }
                })

    def __call__(self):
        print(self._midi_data.keys())
        print(self._midi_data["pops"].keys())
        for k, i in self._midi_data.items():
            files = i["files"]
            for f in files:
                basename = os.path.basename(f)
                yield {
                    "genre": k,
                    "file_name": basename,
                    "piano_roll": i["piano_roll"][basename],
                }

# test code
if __name__=="__main__":
    for loader in MidiLoader("../../../../data/hooktheory")():
        print(loader)
