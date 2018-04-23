import os
import time
from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from six.moves import xrange

from midi_gen_app.analysis.deep_learning.dl_utils.utils import *
from midi_gen_app.analysis.deep_learning.dl_utils.ops import *
from midi_gen_app.analysis.deep_learning.models.midinet import MidiNet

class SmallDataMidiNet(MidiNet):
    def __init__(
        self,
        model_params,
        dl_path,
        sess,
        **data_kwargs
    ):
        self._path = dl_path
        self.sess = sess
        self.base_mels = data_kwargs["base_mels"]
        self.base_prevs = data_kwargs["base_prevs"]
        self.base_chords = data_kwargs["base_chords"]
        self.novel_mels = data_kwargs["novel_mels"]
        self.novel_prevs = data_kwargs["novel_prevs"]
        self.novel_chords = data_kwargs["novel_chords"]
        self.small_mels = data_kwargs["small_mels"] # preparing...
        self.small_prevs = data_kwargs["small_prevs"] # preparing...
        self.small_chords = data_kwargs["small_chords"] # preparing...

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

    def train(self, params):
        self.pretrain(params)

        # fine tuning with novel set + small dataset

        # generate data

    def pretrain(self, params):
        base_data_X = self.base_mels
        base_prev_X = self.base_prevs
        base_data_y = self.base_chords

        base_data_X, base_prev_X, base_data_y = self.extract(
            base_data_X, base_prev_X, base_data_y
        )

        base_data_X, base_prev_X, base_data_y = shuffle(
            base_data_X, base_prev_X, base_data_y, random_state=0
        )

        base_data_X = np.transpose(base_data_X,(0,2,3,1))
        base_prev_X = np.transpose(base_prev_X,(0,2,3,1))

        learning_rate = params["learning_params"]["learning_rate"]
        beta1 = params["learning_params"]["beta1"]

        d_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1
        ).minimize(self.d_loss, var_list=self.d_vars)

        g_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1
        ).minimize(self.g_loss, var_list=self.g_vars)
                                  
        tf.global_variables_initializer().run()

        self.g_sum = tf.summary.merge([
            self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum
        ])
        self.d_sum = tf.summary.merge([
            self.z_sum, self.d_sum, 
            self.d_loss_real_sum, self.d_loss_sum
        ])

        logs_dir = os.path.join(self._path, "logs")
        self.writer = tf.summary.FileWriter(logs_dir, self.sess.graph)

        sample_z = np.random.normal(0, 1, size=(self.sample_size , self.z_dim))
        sample_files = base_data_X[0:self.sample_size]

        sample_dir = os.path.join(self._path, "sample_dir")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)        
        save_images(
            base_data_X[np.arange(len(base_data_X))[:5]]*1, [1, 5],
            os.path.join(sample_dir, 'Train.png')
        )
        checkpoint_dir = os.path.join(self._path, "checkpoint")       
 
        sample_images = base_data_X[0:self.sample_size]
        counter = 0
        start_time = time.time()

        if self.load(checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        sample_labels = sloppy_sample_labels()
        for epoch in xrange(params["pretraining_epoch"]):
            batch_idxs = len(base_data_X) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = \
                    base_data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                prev_batch_images = \
                    base_prev_X[idx*self.batch_size:(idx+1)*self.batch_size]
                
                batch_labels = \
                    base_data_y[idx*self.batch_size:(idx+1)*self.batch_size]
                '''
                Note that the mu and sigma are set to (-1,1) in the experiment of the paper :
                "MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation"
                However, the result are similar by using (0,1)
                '''
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # We've tried to run more d_optim and g_optim, while getting a better result by running g_optim twice in this MidiNet version.
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)
                    
                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels, self.prev_bar:prev_batch_images })
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels })
                errG = self.g_loss.eval({self.images: batch_images, self.z: batch_z, self.y:batch_labels, self.prev_bar:prev_batch_images })
                
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.y:sample_labels, self.prev_bar:prev_batch_images }
                    )
                    filename = 'train_{:02d}_{:04d}.png'.format(epoch, idx)
                    save_images(
                        samples[:5,:], [1, 5],
                        os.path.join(sample_dir, filename)
                    )
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    gen_dir = os.path.join(self._path, "gen")
                    if not os.path.exists(gen_dir):
                        os.makedirs(gen_dir)
                    np_name = 'train_{:02d}_{:04d}'.format(epoch, idx)
                    np_name = os.path.join(gen_dir, np_name)
                    np.save(np_name, samples)

                if np.mod(counter, len(base_data_X)/self.batch_size) == 0:
                    self.save(checkpoint_dir, counter)
            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f" \
            % (epoch, 
                time.time() - start_time, (errD_fake+errD_real)/batch_idxs))

    def fine_tuning(self, params):
        def process_data(X, prev_X, y):
            X, prev_X, y = self.extract(X, prev_X, y)
            X, prev_X, y = shuffle(
                X, prev_X, y, random_state=0
            )
            X = np.transpose(X, (0,2,3,1))
            prev_X = np.transpose(prev_X, (0,2,3,1))
            return X, prev_X, y

        novel_data_X = self.novel_mels
        novel_prev_X = self.novel_prevs
        novel_data_y = self.novel_chords

        novel_data_X, novel_prev_X, novel_data_y = \
            process_data(novel_data_X, novel_prev_X, novel_data_y)

        learning_rate = params["learning_params"]["learning_rate"]
        beta1 = params["learning_params"]["beta1"]

        d_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1
        ).minimize(self.d_loss, var_list=self.d_vars)

        g_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1
        ).minimize(self.g_loss, var_list=self.g_vars)
                                  
        tf.global_variables_initializer().run()
        # small dataset->augmentation by using generation network
        # novel dataset->no generation
        # fine-tuning of discriminator
