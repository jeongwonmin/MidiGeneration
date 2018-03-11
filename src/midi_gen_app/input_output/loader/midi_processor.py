import numpy as np
import pretty_midi as pm


# class for chord calculation
class MidiProcessor(object):
    def __init__(self, loader):
        self._loader = loader

    # fill zeros of chord chroma by extending notes to the next chord
    def _fill_zeros(self, piano):
        # nonzero coordinates along axis 1
        piano_nonzero = np.nonzero(piano)[1]
        # discard invalid file
        if len(piano_nonzero) == 0:
            return None
        piano_nonzero = np.sort(piano_nonzero)
        last = [piano.shape[1]]
        piano_nonzero = np.concatenate([piano_nonzero, last])

        for n in range(0, len(piano_nonzero)-1):
            for i in range(piano_nonzero[n], piano_nonzero[n+1]):
                piano[:,i] = piano[:, piano_nonzero[n]]
        for i in range(0, piano_nonzero[0]):
            piano[:,i] = piano[:, piano_nonzero[0]]
        return piano

    def __call__(self):
        for l in self._loader():
            melody = l["piano_roll"]["melody"]
            chord = l["piano_roll"]["chord"]
            l["piano_roll"]["melody"] = self._fill_zeros(melody)
            l["piano_roll"]["chord"] = self._fill_zeros(chord)
            if l["piano_roll"]["melody"] is None or \
                l["piano_roll"]["chord"] is None:
                continue
            yield l
