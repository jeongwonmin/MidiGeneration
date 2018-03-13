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

    # discard non-basic chords(sus4, dim etc.)
    def _valid_chord(self, chord):
        def check_one_col(mask, col):
            ok = False
            for i in range(12):
                to_mask = col[i:i+len(mask)].astype(int)
                and_ = np.bitwise_and(mask.astype(int), to_mask)
                ok = np.array_equal(mask, and_)
                if ok:
                    break
            return ok

        chord_tile = np.tile(chord, (2, 1))
        mask_major = np.array([1,0,0,0,1,0,0,1])
        mask_minor = np.array([1,0,0,1,0,0,0,1])
        mask_major7 = np.array([1,0,1,0,0,0,1,0,0,1])
        mask_minor7 = np.array([1,0,1,0,0,1,0,0,0,1])
        mask_M7 = np.array([1,1,0,0,0,1,0,0,1])
        
        chk = [
            (check_one_col(mask_major, chord_tile[:,c]) or 
            check_one_col(mask_minor, chord_tile[:,c])) and
            (not check_one_col(mask_major7, chord_tile[:,c]) and
            not check_one_col(mask_minor7, chord_tile[:,c]) and
            not check_one_col(mask_M7, chord_tile[:,c]))
            for c in range(chord_tile.shape[1])
        ]
        return np.all(chk)

    def __call__(self):
        for l in self._loader():
            melody = l["piano_roll"]["melody"]
            chord = l["piano_roll"]["chord"]
            l["piano_roll"]["melody"] = self._fill_zeros(melody)
            l["piano_roll"]["chord"] = self._fill_zeros(chord)
            if l["piano_roll"]["melody"] is None or \
                l["piano_roll"]["chord"] is None:
                continue
            elif not self._valid_chord(l["piano_roll"]["chord"]):
                print(l["file_name"], "contains invalid chord")
                continue
            yield l
