import numpy as np

from .midi_processor import MidiProcessor


class MidiSplitter(object):
    def __init__(self, processor):
        self._processor = processor

    # split into bars
    def _split_into_bars(self, piano_roll):
        melody = piano_roll["melody"]
        chord = piano_roll["chord"]
        
        melody = [
            melody[:,i:min(i+16, melody.shape[1])] 
            for i in range(0, melody.shape[1], 16)
        ]
        print(melody[0].shape, melody[-1].shape)
        if melody[-1].shape != melody[0].shape:
            melody = melody[:-1]

        chord = [
            chord[:, i:min(i+16, chord.shape[1])]
            for i in range(0, chord.shape[1], 16)
        ]
        if chord[-1].shape != chord[0].shape:
            chord = chord[:-1]

        return np.asarray(melody), np.asarray(chord)

    def __call__(self):
        np.set_printoptions(threshold=np.inf)
        for p in self._processor():
            mel, chord = self._split_into_bars(p["piano_roll"])
            p.update({
                "splitted":{
                    "melody": mel,
                    "chord": chord,
                }
            }) 
            yield p

