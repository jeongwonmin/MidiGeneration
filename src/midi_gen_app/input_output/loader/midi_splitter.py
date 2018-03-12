import numpy as np

from .midi_processor import MidiProcessor


class MidiSplitter(object):
    def __init__(self, processor):
        self._processor = processor

    # split into bars
    def _split_into_bars(self, piano_roll):
        def fill_last(last, first):
            first_notes = first.shape[1]
            last_notes = last.shape[1]
            fill = first_notes - last_notes
            fill_last = np.tile(last[:,-1][:,None], (1, fill))
            return np.hstack([last, fill_last])

        melody = piano_roll["melody"]
        chord = piano_roll["chord"]
        
        melody = [
            melody[:,i:min(i+16, melody.shape[1])] 
            for i in range(0, melody.shape[1], 16)
        ]
        if melody[-1].shape != melody[0].shape:
            melody[-1] = fill_last(melody[-1], melody[0])

        chord = [
            chord[:, i:min(i+16, chord.shape[1])]
            for i in range(0, chord.shape[1], 16)
        ]
        if chord[-1].shape != chord[0].shape:
            chord[-1] = fill_last(chord[-1], chord[0])

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

