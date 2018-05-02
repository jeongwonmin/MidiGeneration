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

        """
        bpm_folder = os.path.join(midi_path, "bpms")
        for g in genres:
            bpm_f = g+".csv"
            bpm_f = os.path.join(bpm_folder, bpm_f)
            df = pd.read_csv(bpm_f, header=None)
            df.set_index(0, inplace=True)
            self._midi_data[g].update({"bpms":df})
        """

        self._add_piano_roll()

    def _add_piano_roll(self):
        def fit_chord_to_melody(melody, chord):
            if melody.shape[1] == chord.shape[1]:
                return melody, chord
            elif melody.shape[1] < chord.shape[1]:
                return melody, chord[:,:melody.shape[1]]
            else:
                extra = melody.shape[1] - chord.shape[1]
                zeros = np.zeros((chord.shape[0], extra))
                return melody, np.hstack([chord, zeros])  

		# get piano roll for melody / chroma for chord
        def get_piano_roll(filepath):
            midi_obj = pm.PrettyMIDI(midi_file=filepath)
            insts = midi_obj.instruments
            melody_obj = insts[0]
            chord_obj = insts[1]
            # resolution(tick per quarter) is calculated automatically
            # calculate fs according to the resolution
            fs_ = (1.0/midi_obj.tick_to_time(midi_obj.resolution))*4
            melody = melody_obj.get_piano_roll(fs=fs_)
            chord = chord_obj.get_chroma(fs=fs_)
            melody[melody>0] = 1
            chord[chord>0] = 1
            return fit_chord_to_melody(melody, chord)

        for k, i in self._midi_data.items():
            self._midi_data[k].update({"piano_roll":{}})
            mids = self._midi_data[k]["files"]
            for m in mids:
                basename = os.path.basename(m)
                melody, chord = get_piano_roll(m)
                self._midi_data[k]["piano_roll"].update({
                    basename:{
                        "melody": melody,
                        "chord": chord,
                      }
                })

    def __call__(self):
        for k, i in self._midi_data.items():
            files = i["files"]
            for f in files:
                basename = os.path.basename(f)
                yield {
                    "genre": k,
                    "file_name": basename,
                    "piano_roll": i["piano_roll"][basename],
                }
