import os
from datetime import datetime

import numpy as np

from midi_gen_app.input_output.loader.midi_loader import MidiLoader
from midi_gen_app.input_output.loader.midi_processor import MidiProcessor
from midi_gen_app.input_output.loader.midi_splitter import MidiSplitter
from midi_gen_app.input_output.loader.batch_loader import BatchLoader
from midi_gen_app.input_output.saver.npy_saver import NpySaver
from .models.implementation import Implementation


class DeepLearning(object):
    def __init__(self, **params):
        now_folder = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._path = os.path.join(params["result_dir"], now_folder)

        mels, mel_prevs, chords = None, None, None

        if params["npy_dir"] is None:
            loader = BatchLoader(
                MidiSplitter(MidiProcessor(MidiLoader(params["data_dir"])))
            )
            mels, mel_prevs, chords = loader()
            saver = NpySaver(
                os.path.join(self._path, "npy_data"),
                mels, mel_prevs, chords 
            )
            saver()

        else:
            npys = params["npy_dir"]
            mels_file = os.path.join(npys, "melodies.npy")
            mel_prevs_file = os.path.join(npys, "mel_prevs.npy")
            chords_file = os.path.join(npys, "chords.npy")
            mels = np.load(mels_file)
            mel_prevs = np.load(mel_prevs_file)
            chords = np.load(chords_file)            

        Implementation(params["dl_params"], loader, mels, mel_prevs, chords)

    def __call__(self):
        pass
