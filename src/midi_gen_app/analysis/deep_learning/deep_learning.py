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
from .models.smalldata_midinet import SmallDataMidiNet
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
            self.process_data(params["data_dir"], params["class"])

        else:
            npys = params["npy_dir"]
            mels_file = os.path.join(npys, "melodies.npy")
            mel_prevs_file = os.path.join(npys, "mel_prevs.npy")
            chords_file = os.path.join(npys, "chords.npy")
            self.mels = np.load(mels_file)
            self.mel_prevs = np.load(mel_prevs_file)
            self.chords = np.load(chords_file)            

    def process_data(self, data_dir, dl_class_name):
        self.data_kwargs = {}
        if dl_class_name == "MidiNet":
            loader = BatchLoader(
                MidiSplitter(MidiProcessor(MidiLoader(
                    data_dir,
            ))))
            mels, mel_prevs, chords = loader()
            self.data_kwargs.update({
                "mels": mels,
                "mel_prevs": mel_prevs,
                "chords": chords,
            })
            saver = NpySaver(
                os.path.join(self._path, "npy_data"),
                self.mels, self.mel_prevs, self.chords 
            )
            saver()

        elif dl_class_name == "SmallDataMidiNet":
            base_loader = BatchLoader(
                MidiSplitter(MidiProcessor(MidiLoader(
                    data_dir,
                    genres=["base"]
            ))))

            base_mels, base_prevs, base_chords = base_loader()
            saver = NpySaver(
                os.path.join(self._path, "npy_data", "base"),
                self.mels, self.mel_prevs, self.chords 
            )
            saver()

            self.data_kwargs.update({
                "base_mels": base_mels,
                "base_prevs": base_prevs,
                "base_chords": base_chords,
            })

            novel_loader = BatchLoader(
                MidiSplitter(MidiProcessor(MidiLoader(
                    data_dir,
                    genres=["novel"]
            ))))

            novel_mels, novel_prevs, novel_chords = novel_loader()
            saver = NpySaver(
                os.path.join(self._path, "npy_data", "novel"),
                self.mels, self.mel_prevs, self.chords 
            )
            saver()

            self.data_kwargs.update({
                "novel_mels": novel_mels,
                "novel_prevs": novel_prevs,
                "novel_chords": novel_chords,
            })

            small_mels, small_prevs, small_chords = None, None, None
            self.data_kwargs.update({
                "small_mels": small_mels,
                "small_prevs": small_prevs,
                "small_chords": small_chords,
            })
            # small_data_loader = BatchLoader(
            #         MidiSplitter(MidiProcessor(MidiLoader(
            #         data_dir,
            #         genres=["small"]
            # ))))

    def __call__(self):
        model_class = {
            "MidiNet": MidiNet,
            "SmallDataMidiNet": SmallDataMidiNet,
        }[self.params["class"]]
        with tf.Session() as session:
            print("session start")
            self.model = model_class(
                self.params["model_params"], 
                self._path, 
                **self.data_kwargs, 
                sess=session
            )
            self.model.init_layers()
            self.model(self.params)
        tf.app.run()

