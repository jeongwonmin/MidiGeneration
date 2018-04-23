import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from midi_gen_app.analysis.deep_learning.dl_utils.utils import *
from midi_gen_app.analysis.deep_learning.dl_utils.ops import *


class Implementation(object):
    def __init__(
        self,
        model_params, 
        dl_path,
        sess,
        mels, mel_prevs, chords, 
        ):
        self._path = dl_path
        self.sess = sess
        self.mels = mels
        self.mel_prevs = mel_prevs
        self.chords = chords

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

    def extract(self, mels, mel_prevs, chords):
        mels = mels[:,None,:,:]
        mel_prevs = mel_prevs[:,None,:,:]
        mels = np.transpose(mels, (0,1,3,2)).astype(np.int32)
        mel_prevs = np.transpose(mel_prevs, (0,1,3,2)).astype(np.int32)
        return mels, mel_prevs, chords.astype(np.int32)

    def build_model(self):
        self.saver = tf.train.Saver()

    def train(self, params):
        data_X, prev_X, data_y = self.extract(
            self._mels,
            self._mel_prevs,
            self._chords
        )
    
    def discriminator(self, x, y=None, reuse=False):
        pass

    def generator(self, z, y=None, prev_x=None):
        pass

    def sampler(self, z, y=None, prev_x=None):
        pass

    def save(self, checkpoint_dir, step):
        model_name = self.__class__.__name__+".model"
        model_dir = "%s_%s" % (self.batch_size, self.output_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.batch_size, self.output_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def __call__(self, params):
        self.train(params)
