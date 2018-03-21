import numpy as np


class BatchLoader(object):
    def __init__(self, loader, record=None):
        self._loader = loader
        self.record = record

    def _get_keys_from_pattern(self, pattern, p_list):
        keys = [] 
        for p in p_list:
            if p.find(pattern) != -1:
                keys.append(p)
        return keys   

    def _one_batch(self, one_record):
        f = one_record["file_name"]
        s = one_record["splitted"]
        mel_keys = sorted(set(
            self._get_keys_from_pattern("melody", s.keys())
        ))
        chord_keys = sorted(set(
            self._get_keys_from_pattern("converted", s.keys())
        ))

        if s[mel_keys[0]].shape[0] < 8:
            return

        for mel, chrd in zip(mel_keys, chord_keys):
            shift = mel.split("_")[1]
            key = "_".join([f, shift])
            mel_chrd = [s[mel], s[chrd]]
            yield key, mel_chrd

    def _collect_batch(self, loader):
        if self.record is not None:
            return
        self.record = {}

        mels = []
        mel_prevs = []
        chrds = []

        for l in loader():
            for key, roll in self._one_batch(l):
                if roll[0].shape[0] < 2:
                    continue
                for i in range(roll[0].shape[0]-1):
                    mels.append(roll[0][1:][i])
                    mel_prevs.append(roll[0][:-1][i])
                    chrds.append(roll[1][1:][i])

        return np.asarray(mels), np.asarray(mel_prevs), np.asarray(chrds)

    def __call__(self):
        return self._collect_batch(self._loader)
