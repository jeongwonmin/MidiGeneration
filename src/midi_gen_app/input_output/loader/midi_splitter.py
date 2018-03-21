import numpy as np

from .midi_processor import MidiProcessor


class MidiSplitter(object):
    def __init__(self, processor):
        self._processor = processor

    # split into bars
    def _split_into_bars(self, proll):
        def fill_last(last, first):
            first_notes = first.shape[1]
            last_notes = last.shape[1]
            fill = first_notes - last_notes
            fill_last = np.tile(last[:,-1][:,None], (1, fill))
            return np.hstack([last, fill_last])

        proll_bars = [
            proll[:,i:min(i+16, proll.shape[1])] 
            for i in range(0, proll.shape[1], 16)
        ]

        if proll_bars[-1].shape != proll_bars[0].shape:
            proll_bars[-1] = fill_last(proll_bars[-1], proll_bars[0])

        return np.asarray(proll_bars)

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
            # chord_dim13 = np.tile(chord_dim13[:,None], (1,16))
            splitted.append(chord_dim13)
        return np.asarray(splitted)

    def _get_keys_from_pattern(self, pattern, p_list):
        keys = [] 
        for p in p_list:
            if p.find(pattern) != -1:
                keys.append(p)
        return keys   

    def __call__(self):
        np.set_printoptions(threshold=np.inf)
        for p in self._processor():
            a = p["piano_roll"]["adjusted"]
            mel_keys = self._get_keys_from_pattern("melody", a.keys())
            chord_keys = self._get_keys_from_pattern("chord", a.keys())

            for mk in mel_keys:
                splitted_m = self._split_into_bars(a[mk])
                if "splitted" not in p.keys():
                    p.update({"splitted": {}})
                p["splitted"].update({mk: splitted_m})

            for ck in chord_keys:
                splitted_c = self._split_into_bars(a[ck])
                p["splitted"].update({
                    ck: splitted_c,
                    "_".join([ck, "converted"]): \
                        self._convert_chord(splitted_c),
                    })
            yield p

