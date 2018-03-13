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

    # get shift size to adjust melody between C4 and B5
    def _adjust_melody_c4b5(self, melody):
        melody_range = np.sort(np.unique(np.nonzero(melody)[0]))
        max_note = melody_range[-1]
        min_note = melody_range[0]
        gap = max_note - min_note
        if gap >= 24:
            return None

        c4 = 48
        b5 = 71
        if min_note >= c4 and max_note <= b5:
            return 0
        elif max_note <= c4:
            return (c4 - max_note) + gap + 1
        elif min_note >= b5:
            return ((min_note - b5) + gap + 1) * (-1)
        elif min_note < c4 and max_note <= b5:
            return c4 - min_note
        elif min_note > c4 and max_note >= b5:
            return (-1) * (max_note - b5)
            
    def __call__(self):
        count = 0
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
            elif self._adjust_melody_c4b5(l["piano_roll"]["melody"]) is None:
                continue
            r = self._adjust_melody_c4b5(l["piano_roll"]["melody"])
            l["piano_roll"]["melody"] = np.roll(
                l["piano_roll"]["melody"], r, axis=0
            )
            l["piano_roll"]["chord"] = np.roll(
                l["piano_roll"]["chord"], r, axis=0
            )
            yield l
            count += 1
        print(count)
