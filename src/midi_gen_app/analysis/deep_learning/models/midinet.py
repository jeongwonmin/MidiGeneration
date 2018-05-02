import os
import time
from glob import glob

import tensorflow as tf
from sklearn.utils import shuffle
from six.moves import xrange

from .implementation import Implementation
from midi_gen_app.analysis.deep_learning.dl_utils.utils import *
from midi_gen_app.analysis.deep_learning.dl_utils.ops import *


class MidiNet(Implementation):
    def init_layers(self):
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn0 = batch_norm(name='d_bn0')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        if self.prev_dim:
            self.g_prev_bn0 = batch_norm(name='g_prev_bn0')
            self.g_prev_bn1 = batch_norm(name='g_prev_bn1')
            self.g_prev_bn2 = batch_norm(name='g_prev_bn2')
            self.g_prev_bn3 = batch_norm(name='g_prev_bn3')


        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.build_model()

    def build_model(self):
        self.y= tf.placeholder(
            tf.float32, [self.batch_size, self.y_dim], name='y'
        )

        self.prev_bar = tf.placeholder(
            tf.float32, 
            [self.batch_size] + [self.output_w, self.output_h, self.c_dim],
            name='prev_bar'
        )
        self.images = tf.placeholder(
            tf.float32, 
            [self.batch_size] + [self.output_w, self.output_h, self.c_dim],
            name='real_images'
        )
        self.sample_images= tf.placeholder(
            tf.float32, 
            [self.sample_size] + [self.output_w, self.output_h, self.c_dim],
            name='sample_images'
        )
        self.z = tf.placeholder(
            tf.float32, 
            [None, self.z_dim],
            name='z'
        )
        self.z_sum = tf.summary.histogram("z", self.z)
        self.G = self.generator(self.z, self.y, self.prev_bar)
        self.D, self.D_logits, self.fm = self.discriminator(
            self.images, self.y, reuse=False
        )

        self.sampler = self.sampler(self.z, self.y, self.prev_bar)
        self.D_, self.D_logits_ , self.fm_ = self.discriminator(
            self.G, self.y, reuse=True
        )

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=0.9*tf.ones_like(self.D)
        ))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)
        ))
        self.g_loss0 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)
        ))

        #Feature Matching
        self.features_from_g = tf.reduce_mean(self.fm_, reduction_indices=(0))
        self.features_from_i = tf.reduce_mean(self.fm, reduction_indices=(0))
        self.fm_g_loss1 =tf.multiply(
            tf.nn.l2_loss(self.features_from_g - self.features_from_i), 0.1
        )

        self.mean_image_from_g = tf.reduce_mean(self.G, reduction_indices=(0))
        self.mean_image_from_i = tf.reduce_mean(
            self.images, reduction_indices=(0)
        )
        self.fm_g_loss2 = tf.multiply(
            tf.nn.l2_loss(
                self.mean_image_from_g - self.mean_image_from_i
            ), 0.01
        )

        self.d_loss_real_sum = tf.summary.scalar(
            "d_loss_real", self.d_loss_real
        )
        self.d_loss_fake_sum = tf.summary.scalar(
            "d_loss_fake", self.d_loss_fake
        )

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = self.g_loss0 + self.fm_g_loss1 + self.fm_g_loss2

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, params):
        data_X = self.mels
        prev_X = self.mel_prevs
        data_y = self.chords

        data_X, prev_X, data_y = self.extract(data_X, prev_X, data_y)

        data_X, prev_X, data_y = shuffle(data_X,prev_X,data_y, random_state=0)
        
        data_X = np.transpose(data_X,(0,2,3,1))
        prev_X = np.transpose(prev_X,(0,2,3,1))

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
        sample_files = data_X[0:self.sample_size]

        sample_dir = os.path.join(self._path, "sample_dir")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)        
        save_images(
            data_X[np.arange(len(data_X))[:5]]*1, [1, 5],
            os.path.join(sample_dir, 'Train.png')
        )
        checkpoint_dir = os.path.join(self._path, "checkpoint")       
 
        sample_images = data_X[0:self.sample_size]
        counter = 0
        start_time = time.time()

        if self.load(checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        sample_labels = sloppy_sample_labels()
        for epoch in xrange(params["epoch"]):
            batch_idxs = len(data_X) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = \
                    data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                prev_batch_images = \
                    prev_X[idx*self.batch_size:(idx+1)*self.batch_size]
                
                batch_labels = \
                    data_y[idx*self.batch_size:(idx+1)*self.batch_size]
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

                if np.mod(counter, len(data_X)/self.batch_size) == 0:
                    self.save(checkpoint_dir, counter)
            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f" \
            % (epoch, 
                time.time() - start_time, (errD_fake+errD_real)/batch_idxs))
 
    def discriminator(self, x, y=None, reuse=False):
        df_dim = self.df_dim
        dfc_dim = self.dfc_dim

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(
                    self.d_bn0(conv2d(x, 64, k_h=4, k_w=89, name='d_h0_conv')))
                h1 = lrelu(
                    self.d_bn1(conv2d(h0, 64, k_h=4, k_w=1, name='d_h1_conv')))
                h2 = lrelu(
                    self.d_bn2(conv2d(h1, 64, k_h=4, k_w=1, name='d_h2_conv')))
                h3 = linear(
                    tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin'
                )

                return tf.nn.sigmoid(h3), h3
      
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(x, yb)
                h0 = lrelu(
                    conv2d(
                        x, self.c_dim + self.y_dim,
                        k_h=2, k_w=128, name='d_h0_conv'
                    )
                )
                fm = h0
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(
                    self.d_bn1(
                        conv2d(
                            h0, self.df_dim + self.y_dim,
                            k_h=4, k_w=1, name='d_h1_conv'
                    ))
                )
                h1 = tf.reshape(h1, [self.batch_size, -1])            
                h1 = tf.concat([h1, y], 1)
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')
                
                return tf.nn.sigmoid(h3), h3, fm

    def generator(self, z, y=None, prev_x = None, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            h0_prev = lrelu(self.g_prev_bn0(conv2d(
                prev_x, 16, k_h=1, k_w=128,d_h=1, d_w=2, name='g_h0_prev_conv'
            )))
            h1_prev = lrelu(self.g_prev_bn1(conv2d(
                h0_prev, 16, k_h=2, k_w=1, name='g_h1_prev_conv'
            )))
            h2_prev = lrelu(self.g_prev_bn2(conv2d(
                h1_prev, 16, k_h=2, k_w=1, name='g_h2_prev_conv'
            )))
            h3_prev = lrelu(self.g_prev_bn3(conv2d(
                h2_prev, 16, k_h=2, k_w=1, name='g_h3_prev_conv'
            )))
           
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, 1024, 'g_h0_lin')))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(
                self.g_bn1(linear(h0, self.gf_dim*2*2*1, 'g_h1_lin'))
            )

            h1 = tf.reshape(h1, [self.batch_size, 2, 1, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)
            h1 = conv_prev_concat(h1, h3_prev)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(
                h1, [self.batch_size, 4, 1, self.gf_dim * 2],
                k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h2'
            )))
            h2 = conv_cond_concat(h2, yb)
            h2 = conv_prev_concat(h2, h2_prev)

            h3 = tf.nn.relu(self.g_bn3(deconv2d(
                h2, [self.batch_size, 8, 1, self.gf_dim * 2],
                k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h3'
            )))
            h3 = conv_cond_concat(h3, yb)
            h3 = conv_prev_concat(h3, h1_prev)

            h4 = tf.nn.relu(self.g_bn4(deconv2d(
                h3, [self.batch_size, 16, 1, self.gf_dim * 2],
                k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h4'
            )))
            h4 = conv_cond_concat(h4, yb)
            h4 = conv_prev_concat(h4, h0_prev)

            return tf.nn.sigmoid(deconv2d(
                h4, [self.batch_size, 16, 128, self.c_dim],
                k_h=1, k_w=128,d_h=1, d_w=2, name='g_h5'
            ))

    def sampler(self, z, y=None, prev_x=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            h0_prev = lrelu(self.g_prev_bn0(conv2d(
                prev_x, 16, k_h=1, k_w=128, d_h=1, d_w=2,name='g_h0_prev_conv'
            )))
            h1_prev = lrelu(self.g_prev_bn1(conv2d(
                h0_prev, 16, k_h=2, k_w=1, name='g_h1_prev_conv'
            )))
            h2_prev = lrelu(self.g_prev_bn2(conv2d(
                h1_prev, 16, k_h=2, k_w=1, name='g_h2_prev_conv'
            )))
            h3_prev = lrelu(self.g_prev_bn3(conv2d(
                h2_prev, 16, k_h=2, k_w=1, name='g_h3_prev_conv'
            )))
            
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(self.g_bn0(linear(z, 1024, 'g_h0_lin')))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*2*1, 'g_h1_lin')))

            h1 = tf.reshape(h1, [self.batch_size, 2, 1, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)
            h1 = conv_prev_concat(h1, h3_prev)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(
                h1, [self.batch_size, 4, 1, self.gf_dim * 2],
                k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h2'
            )))
            h2 = conv_cond_concat(h2, yb)
            h2 = conv_prev_concat(h2, h2_prev)

            h3 = tf.nn.relu(self.g_bn3(deconv2d(
                h2, [self.batch_size, 8, 1, self.gf_dim * 2],
                k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h3'
            )))
            h3 = conv_cond_concat(h3, yb)
            h3 = conv_prev_concat(h3, h1_prev)

            h4 = tf.nn.relu(self.g_bn4(deconv2d(
                h3, [self.batch_size, 16, 1, self.gf_dim * 2],
                k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h4'
            )))
            h4 = conv_cond_concat(h4, yb)
            h4 = conv_prev_concat(h4, h0_prev)

            return tf.nn.sigmoid(deconv2d(
                h4, [self.batch_size, 16, 128, self.c_dim],
                k_h=1, k_w=128,d_h=1, d_w=2, name='g_h5'
            ))
       
