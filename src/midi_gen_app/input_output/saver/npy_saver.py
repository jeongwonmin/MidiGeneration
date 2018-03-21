import os
import numpy as np


class NpySaver(object):
    def __init__(self, save_path, mels, mel_prevs, chrds):
        self._mels, self._mel_prevs, self._chrds = mels, mel_prevs, chrds
        self._path = save_path
        if not os.path.exists(self._path):
            os.makedirs(self._path)
 
    def __call__(self):
        def _path(filename):
            return os.path.join(self._path, filename)

        print("saving data")
        np.save(_path("melodies.npy"), self._mels)
        np.save(_path("mel_prevs.npy"), self._mel_prevs)
        np.save(_path("chords.npy"), self._chrds)
