import os
from datetime import datetime

import numpy as np

from midi_gen_app.input_output.loader.midi_loader import MidiLoader
from midi_gen_app.input_output.loader.midi_processor import MidiProcessor
from midi_gen_app.input_output.loader.midi_splitter import MidiSplitter
from midi_gen_app.input_output.loader.batch_loader import BatchLoader
from midi_gen_app.input_output.saver.npy_saver import NpySaver
from .models.implementation import Implementation
from .models.midinet import MidiNet
import tensorflow as tf


class DeepLearning(object):
    def __init__(self, **params):
        self.params = params
        now_folder = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._path = os.path.join(params["result_dir"], now_folder)

        self.mels, self.mel_prevs, self.chords = None, None, None

        # mels shape: (n, 128, 16), mel_prevs shape: (n, 128, 16),
        # chords shape: (n, 13)
        # n: bar number
        # prev bar of the first bar is np.zeros((128, 16))
        if params["npy_dir"] is None:
            loader = BatchLoader(
                MidiSplitter(MidiProcessor(MidiLoader(params["data_dir"])))
            )
            self.mels, self.mel_prevs, self.chords = loader()
            saver = NpySaver(
                os.path.join(self._path, "npy_data"),
                self.mels, self.mel_prevs, self.chords 
            )
            saver()

        else:
            npys = params["npy_dir"]
            mels_file = os.path.join(npys, "melodies.npy")
            mel_prevs_file = os.path.join(npys, "mel_prevs.npy")
            chords_file = os.path.join(npys, "chords.npy")
            self.mels = np.load(mels_file)
            self.mel_prevs = np.load(mel_prevs_file)
            self.chords = np.load(chords_file)            

    def __call__(self):
        with tf.Session() as sess:
            self.model = MidiNet(self.params["model_params"], 
                self._path, self.mels, self.mel_prevs, self.chords, sess
            )
            self.model.init_layers()
           
            self.model(self.params)
        tf.app.run()
