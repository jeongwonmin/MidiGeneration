import numpy as np
import pretty_midi as pm


# class for chord calculation
class MidiProcessor(object):
    def __init__(self, loader):
        self._loader = loader

    # fill zeros of chord chroma by extending notes to the next chord
    def _fill_zeros(self, chord):
        # nonzero coordinates along axis 1
        chord_nonzero = np.nonzero(chord)[1]
        chord_nonzero = np.sort(chord_nonzero)
        last = [chord.shape[1]]
        chord_nonzero = np.concatenate([chord_nonzero, last])
        for n in range(0, len(chord_nonzero)-1):
            for i in range(chord_nonzero[n], chord_nonzero[n+1]):
                chord[:,i] = chord[:, chord_nonzero[n]]
        return chord


    def __call__(self):
        for l in self._loader():
            chord = l["piano_roll"]["chord"]
            l["piano_roll"]["chord"] = self._fill_zeros(chord)
            yield l
