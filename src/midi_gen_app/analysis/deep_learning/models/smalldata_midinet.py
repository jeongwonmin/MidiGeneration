import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from midi_gen_app.analysis.deep_learning.dl_utils.utils import *
from midi_gen_app.analysis.deep_learning.dl_utils.ops import *
from midi_gen_app.analysis.deep_learning.models.midinet import MidiNet

class SmallDataMidiNet(MidiNet):
    def __init__(
        self,
        model_params,
        dl_path,
        data_kwargs, # dict type variables
        sess=None
    ):
        self._path = dl_path
        self.sess = sess
        self.base_mels = base_mels
        self.base_prevs = base_prevs
        self.base_chords = base_chords
        self.novel_mels = novel_mels
        self.novel_prevs = novel_prevs
        self.novel_chords = novel_chords
        self.small_mels = small_mels # preparing...
        self.small_prevs = small_prevs # preparing...
        self.small_chords = small_chords # preparing...

        self.batch_size = model_params["batch_size"]
        self.sample_size = model_params["sample_size"]
        self.output_w = model_params["output_w"]
        self.output_h = model_params["output_h"]
        self.y_dim = model_params["y_dim"]
        self.prev_dim = model_params["prev_dim"]
        self.z_dim = model_params["z_dim"]
        self.gf_dim = model_params["gf_dim"]
        self.df_dim = model_params["df_dim"]
        self.gfc_dim = model_params["gfc_dim"]
        self.dfc_dim = model_params["dfc_dim"]
        self.c_dim = model_params["c_dim"]

