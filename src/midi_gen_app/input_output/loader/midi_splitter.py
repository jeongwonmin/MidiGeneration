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

    # convert major/minor chord to 13-dim chord(in MidiNet paper)
    # shape is (13, 16) -> 13: chord-dim, 16: from minimum note length 1/16
    def _convert_chord(self, chord_splitted):
        def convert_to_13dim(major_mask, minor_mask, col):
            major_base, ismajor = None, False
            minor_base, isminor = None, False
            for i in range(12):
                to_mask = col[i:i+len(major_mask)].astype(int)
                major = np.bitwise_and(major_mask.astype(int), to_mask)
                ismajor = np.array_equal(major_mask, major)
                major_base = i

                minor = np.bitwise_and(minor_mask.astype(int), to_mask)
                isminor = np.array_equal(minor_mask, minor)
                minor_base = i
                if ismajor:
                    dim_13 = np.zeros(13, dtype=int)
                    dim_13[major_base] = 1
                    dim_13[-1] = not ismajor
                    return dim_13
                elif isminor:
                    dim_13 = np.zeros(13, dtype=int)
                    dim_13[minor_base] = 1
                    # start from A
                    dim_13[:-1] = np.roll(dim_13[:-1], 3) 
                    dim_13[-1] = isminor
                    return dim_13

        splitted = []
        for i in range(chord_splitted.shape[0]):
            chord_tile = np.tile(chord_splitted[i], (2,1))
            mask_major = np.array([1,0,0,0,1,0,0,1])
            mask_minor = np.array([1,0,0,1,0,0,0,1])
            chord_dim13 = convert_to_13dim(
                mask_major, mask_minor, chord_tile[:,0]
            )
            chord_dim13 = np.tile(chord_dim13[:,None], (1,16))
            splitted.append(chord_dim13)
        return np.asarray(splitted)

    def __call__(self):
        np.set_printoptions(threshold=np.inf)
        for p in self._processor():
            mel, chord = self._split_into_bars(p["piano_roll"])
            p.update({
                "splitted":{
                    "melody": mel,
                    "chord": chord,
                    "chord_converted": self._convert_chord(chord),
                }
            }) 
            yield p

