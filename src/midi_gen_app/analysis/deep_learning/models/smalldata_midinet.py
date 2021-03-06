import os
import time
from glob import glob
from copy import deepcopy

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
        self.task = model_params["task"]

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
        self.small_rate = model_params["small_rate"]
        self.pretrained_model = model_params["pretrained_model"]
        self.threshold = model_params["threshold"]
        self.alpha = model_params["alpha_blend"] # alpha blend(boosting feature and generated melody)
        self.use_fake = model_params["use_fake"]
        self.task = model_params["task"]

    def train(self, params):
        self.pretrain(params)
        self.fine_tuning(params)
        # fine tuning with novel set + small dataset

        # generate data

    def generate_melody(self, params):
        def process_data(X, prev_X, y):
            X, prev_X, y = self.extract(X, prev_X, y)
            X = np.transpose(X, (0,2,3,1))
            prev_X = np.transpose(prev_X, (0,2,3,1))
            return X, prev_X, y

        logs_dir = os.path.join(self._path, "logs")
        self.writer = tf.summary.FileWriter(logs_dir, self.sess.graph)

        #tf.global_variables_initializer().run()
        self.load(self.pretrained_model)
        print("Load SUCCESS")

        novel_data_X = self.novel_mels
        novel_prev_X = self.novel_prevs
        novel_data_y = self.novel_chords
        small_data_X = self.small_mels
        small_prev_X = self.small_prevs
        small_data_y = self.small_chords

        novel_data_X, novel_prev_X, novel_data_y = \
            process_data(novel_data_X, novel_prev_X, novel_data_y)

        small_data_X, small_prev_X, small_data_y = \
            process_data(small_data_X, small_prev_X, small_data_y)

        small_data_X, small_prev_X, small_data_y = \
            shuffle(
                small_data_X, small_prev_X, small_data_y, random_state=1
            )
        # novel: 正しくはfeature boost用の音楽
        avgs = average_melodies(novel_data_X, novel_data_y)
        avg_np = np.array([a[1] for a in avgs])
        avg_folder = os.path.join(self._path, "average")
        if not os.path.exists(avg_folder):
            os.makedirs(avg_folder)
        np.save(os.path.join(avg_folder, "average.npy"), avg_np) 

        gen_small, gen_boosted, gen_y = \
            self._make_fake_melody(
                small_data_X,
                small_prev_X,
                small_data_y,
                avgs,
                fake_num=8*2,
                save_folder="generated_melody"
            )

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

        self.d_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1
        ).minimize(self.d_loss, var_list=self.d_vars)

        self.g_optim = tf.train.AdamOptimizer(
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
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
 
        sample_images = base_data_X[0:self.sample_size]
        counter = 0
        start_time = time.time()

        # if self.load(checkpoint_dir):
        if self.pretrained_model is not None:
            if self.load(self.pretrained_model):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
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
                
                # Update D network / save d_w for fine tuning
                self.d_w, summary_str = self.sess.run([self.d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                self.g_w, summary_str = self.sess.run([self.g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # We've tried to run more d_optim and g_optim, while getting a better result by running g_optim twice in this MidiNet version.
                self.g_w, summary_str = self.sess.run([self.g_optim, self.g_sum],
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

                if epoch % 5 == 0 or epoch == params["pretraining_epoch"] - 1:
                    self.save(checkpoint_dir, epoch)
                print("Epoch: [%2d] time: %4.4f, d_loss: %.8f" \
                % (epoch, 
                    time.time() - start_time, (errD_fake+errD_real)/batch_idxs))

    def fine_tuning(self, params):
        def process_data(X, prev_X, y):
            X, prev_X, y = self.extract(X, prev_X, y)
            X = np.transpose(X, (0,2,3,1))
            prev_X = np.transpose(prev_X, (0,2,3,1))
            return X, prev_X, y

        novel_data_X = self.novel_mels
        novel_prev_X = self.novel_prevs
        novel_data_y = self.novel_chords
        small_data_X = self.small_mels
        small_prev_X = self.small_prevs
        small_data_y = self.small_chords

        novel_data_X, novel_prev_X, novel_data_y = \
            process_data(novel_data_X, novel_prev_X, novel_data_y)

        small_data_X, small_prev_X, small_data_y = \
            process_data(small_data_X, small_prev_X, small_data_y)

        # novel: 正しくはfeature boost用の音楽
        avgs = average_melodies(novel_data_X, novel_data_y)
        avg_np = np.array([a[1] for a in avgs])
        avg_folder = os.path.join(self._path, "average")
        if not os.path.exists(avg_folder):
            os.makedirs(avg_folder)
        np.save(os.path.join(avg_folder, "average.npy"), avg_np) 

        learning_rate = params["learning_params"]["learning_rate"]
        beta1 = params["learning_params"]["beta1"]

        # small batches (novel:small = small_rate:1)
        # if you don't use novel set, set small_rate=0

        small_batch_n = 15000
        if self.use_fake:
            real = (small_batch_n // (self.small_rate+1)) * self.small_rate
            fake = small_batch_n - real

        else:
            real = small_batch_n
            fake = 0

        # initialize with pretrained model                          
        # tf.global_variables_initializer().run(self.d_w)
        # tf.global_variables_initializer().run(self.g_w)

        gen_small, gen_boosted, gen_y = \
            self._make_fake_melody(
                small_data_X,
                small_prev_X,
                small_data_y,
                avgs,
                fake
            )
        # convert melodies to binary image
        gen_small = binary_melodies(gen_small)
        gen_boosted = binary_melodies(gen_boosted)

        gen_prev = np.concatenate((
            np.zeros(gen_small[0].shape)[None,:,:,:],
            gen_small[:gen_small.shape[0]-1]
        ))

        gen_boosted_prev = np.concatenate((
            np.zeros(gen_boosted[0].shape)[None,:,:,:],
            gen_boosted[:gen_boosted.shape[0]-1]
        ))

        fake_mel_folder = os.path.join(self._path, "fake_melody")

        np.save(os.path.join(fake_mel_folder, "generated.npy"), gen_small)
        np.save(os.path.join(fake_mel_folder, "boosted.npy"), gen_boosted)

        small_len = len(small_data_X)
        small_repeat = real // small_len
        small_data_X = np.tile(small_data_X, (small_repeat, 1,1,1))
        small_prev_X = np.tile(small_prev_X, (small_repeat, 1,1,1))
        small_data_y = np.tile(small_data_y, (small_repeat, 1))

        if self.use_fake:
            novel_small_X = np.r_[small_data_X, gen_small, gen_boosted]
            novel_small_p = np.r_[small_prev_X, gen_prev, gen_boosted_prev] # prev: preparing...
            novel_small_y = np.r_[small_data_y, gen_y, gen_y]

        else:
            novel_small_X = small_data_X
            novel_small_p = small_prev_X
            novel_small_y = small_data_y

        novel_small_X, novel_small_p, novel_small_y = \
            shuffle(
                novel_small_X, novel_small_p, novel_small_y, random_state=1
            )

        sample_labels = sloppy_sample_labels()
        counter=0
        start_time = time.time()
        sample_z = np.random.normal(0, 1, size=(self.sample_size , self.z_dim))
        sample_images = novel_small_X[0:self.sample_size]
        sample_dir = os.path.join(self._path, "fine_sample_dir")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        save_images(
            novel_small_X[np.arange(len(novel_small_X))[:5]]*1, [1, 5],
            os.path.join(sample_dir, 'fine_Train.png')
        )
        checkpoint_dir = os.path.join(self._path, "fine_checkpoint")       
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for epoch in xrange(params["fine_tuning_epoch"]):
            batch_idxs = len(novel_small_X) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = \
                    novel_small_X[idx*self.batch_size:(idx+1)*self.batch_size]
                prev_batch_images = \
                    novel_small_p[idx*self.batch_size:(idx+1)*self.batch_size]
                
                batch_labels = \
                    novel_small_y[idx*self.batch_size:(idx+1)*self.batch_size]
                '''
                Note that the mu and sigma are set to (-1,1) in the experiment of the paper :
                "MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation"
                However, the result are similar by using (0,1)
                '''
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
                
                # Update D network / save d_w for fine tuning
                self.d_w, summary_str = self.sess.run([self.d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                self.g_w, summary_str = self.sess.run([self.g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z ,self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # We've tried to run more d_optim and g_optim, while getting a better result by running g_optim twice in this MidiNet version.
                self.g_w, summary_str = self.sess.run([self.g_optim, self.g_sum],
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
                    filename = 'fine_train_{:02d}_{:04d}.png'.format(epoch, idx)
                    save_images(
                        samples[:5,:], [1, 5],
                        os.path.join(sample_dir, filename)
                    )
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    gen_dir = os.path.join(self._path, "fine_gen")
                    if not os.path.exists(gen_dir):
                        os.makedirs(gen_dir)
                    np_name = 'fine_train_{:02d}_{:04d}'.format(epoch, idx)
                    np_name = os.path.join(gen_dir, np_name)
                    np.save(np_name, samples)

                    label_name = np_name + "_label"
                    label_name = os.path.join(gen_dir, label_name)
                    np.save(label_name, sample_labels)

                if np.mod(counter, batch_idxs) == 1:
                    self.save(checkpoint_dir, counter)
            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f" \
            % (epoch, 
                time.time() - start_time, (errD_fake+errD_real)/batch_idxs))


        # small dataset->augmentation by using generation network
        # novel dataset->no generation
        # fine-tuning of discriminator

        print("fine tuning ended")

    def _make_fake_melody(
        self, 
        X, prev_X, data_y, avgs, 
        fake_num, save_folder="fake_melody"
    ):
        # generate from small dataset (only use filtered generation results)
        gen_passed = []
        gen_y = []
        gen_failed = [] # melodies which couldn't passed discriminator
        gen_failed_y = [] # labels for gen_failed

        # generate by using small data -> filter by discriminator
        batch_idxs = len(X) // self.batch_size
        seed = 10000000
        thres = self.threshold # sigmoid threshold
        while len(gen_passed) <= fake_num // 2:
            i = len(gen_passed)
            if i % 20 == 0:
                print("small data length: {} / {}".format(i, fake_num))
            for idx in xrange(0, batch_idxs):
                batch_images = \
                    X[idx*self.batch_size:(idx+1)*self.batch_size]
                prev_batch_images = \
                    prev_X[idx*self.batch_size:(idx+1)*self.batch_size]
                
                batch_labels = \
                    data_y[idx*self.batch_size:(idx+1)*self.batch_size]
                '''
                Note that the mu and sigma are set to (-1,1) in the experiment of the paper :
                "MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation"
                However, the result are similar by using (0,1)
                '''
                np.random.seed(seed)
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
                seed += 1

                z = tf.convert_to_tensor(batch_z, np.float32)
                y = tf.convert_to_tensor(batch_labels, np.float32)
                prev = tf.convert_to_tensor(prev_batch_images, np.float32)
                gen = self.generator(z, y, prev, reuse=True)

                # if thres == 0, do not pass through the discriminator
                if thres == 0:
                    gen_np = gen.eval()
                    gen_passed = gen_passed + gen_np.tolist()
                    gen_y = gen_y + batch_labels.tolist()

                else:
                # filter with discriminator
                    d, d_logits, fm = self.discriminator(
                        gen, y, reuse=True
                    )

                    d_np = d.eval()
                    gen_np = gen.eval()
                    # 8小節ごとに切って「直前の小説」を取っておく
                    filter_r = np.where(d_np>thres)[0]
                    gen_passed = gen_passed + gen_np[filter_r,:,:,:].tolist()
                    gen_y = gen_y + batch_labels[filter_r,:].tolist()

                    filter_fail = np.where(d_np <= thres)[0]
                    gen_failed = gen_failed + gen_np[filter_fail,:,:,:].tolist()
                    gen_failed_y = gen_failed_y \
                        + batch_labels[filter_fail,:].tolist()

        gen_passed = np.array(gen_passed)
        gen_y = np.array(gen_y)
        gen_boosted = deepcopy(gen_passed)
        for c, a in avgs:
            idx = np.where(np.all(gen_y==c, axis=1))[0]
            gen_boosted[idx] = self.alpha * gen_boosted[idx] \
                + (1 - self.alpha) * a

        fake_mel_folder = os.path.join(self._path, save_folder)
        if not os.path.exists(fake_mel_folder):
            os.makedirs(fake_mel_folder)

        np.save(
            os.path.join(fake_mel_folder, "generated_original.npy"),
            gen_passed
        )
        np.save(
            os.path.join(fake_mel_folder, "boosted_original.npy"),
            gen_boosted
        )
        np.save(
            os.path.join(fake_mel_folder, "labels.npy"),
            gen_y
        )
        
        if len(gen_failed) > 0:
            gen_failed = np.array(gen_failed)
            gen_failed_y = np.array(gen_failed_y)
            np.save(
                os.path.join(fake_mel_folder, "generated_failed.npy"),
                gen_failed
            )
            np.save(
                os.path.join(fake_mel_folder, "labels_failed.npy"),
                gen_failed_y
            )

        return gen_passed, gen_boosted,  gen_y

    def __call__(self, params):
        if self.task == "train":
            self.train(params)
        elif self.task == "generation":
            self.generate_melody(params)
        
