
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import random
import time
import scipy.misc
import utils

version = 'faces'

newPoke_path = './' + version

random_dim = 64
height, width, channel = 64, 64, 3
batch_size = 64
gp_lambda = 10
learning_rate = 0.0001
beta1 = 0.5
beta2 = 0.9


def preprocess_fn(img):
    re_size = 64
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/faces/*.jpg')
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[96, 96, 3], preprocess_fn=preprocess_fn)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x)


def generator(z, is_train=True, reuse=True):
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    with tf.variable_scope('generator') as scope:
        # z = slim.fully_connected(z, 4 * 4 * 512, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu, reuse=reuse,
        #                            scope='g_project', weights_initializer=initializer)
        z = tf.reshape(z, [-1, 32, 32, 4])


        z = slim.convolution2d(z, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding='same',
                               normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                                            activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)
        # z: [32, 32]

        z = slim.convolution2d(z, num_outputs=64, kernel_size=[3, 3], stride=[2, 2], padding='same',
                               normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                                            activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer)
        # z: [16, 16]

        z = slim.convolution2d(z, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], padding='same',
                               normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                               activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer)
        # z: [16, 16]
        z = slim.convolution2d(z, num_outputs=256, kernel_size=[3, 3], stride=[2, 2], padding='same',
                               normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                               activation_fn=tf.nn.relu, scope='g_conv4', weights_initializer=initializer)
        # z: [8, 8]

        z = slim.convolution2d(z, num_outputs=512, kernel_size=[3, 3], stride=[2, 2], padding='same',
                               normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                               activation_fn=tf.nn.relu, scope='g_conv5', weights_initializer=initializer)
        # z: [4, 4]

        z = slim.convolution2d_transpose(z, num_outputs=256, kernel_size=[3, 3], stride=[2, 2], padding="same",
                                            normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                                            activation_fn=tf.nn.relu, scope='g_trans_conv1', weights_initializer=initializer)
        # z: [8, 8]

        z = slim.convolution2d_transpose(z, num_outputs=128, kernel_size=[3, 3], stride=[2, 2], padding="same",
                                            normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                                            activation_fn=tf.nn.relu, scope='g_trans_conv2', weights_initializer=initializer)
        # z: [16, 16]

        z = slim.convolution2d_transpose(z, num_outputs=64, kernel_size=[3, 3], stride=[2, 2], padding="same",
                                            normalizer_fn=slim.batch_norm, trainable=is_train, reuse=reuse,
                                            activation_fn=tf.nn.relu, scope='g_trans_conv3', weights_initializer=initializer)
        # z: [32, 32]


        z = slim.convolution2d_transpose(z, num_outputs=3,  kernel_size=[3, 3], stride=[2, 2], padding="same", reuse=reuse,
                                             biases_initializer=None, activation_fn=tf.nn.tanh, trainable=is_train,
                                             scope='g_out', weights_initializer=initializer)
        # z: [64, 64]

    return z


def discriminator(input, reuse=True):
    initializer = tf.truncated_normal_initializer(stddev=0.02)
    with tf.variable_scope('discriminator') as scope:


        # z: [64, 64]

        input = slim.convolution2d(input, 32, [3, 3], stride=[1, 1], padding="same", biases_initializer=None,
                                  activation_fn=lrelu, reuse=reuse, scope='d_conv1', weights_initializer=initializer)
        # z: [64, 64]

        input = slim.convolution2d(input, 64, [3, 3], stride=[2, 2], padding="same", biases_initializer=None,
                                  activation_fn=lrelu, reuse=reuse, scope='d_conv2', weights_initializer=initializer)
        # z: [32, 32]

        input = slim.convolution2d(input, 128, [3, 3], stride=[1, 1], padding="same", biases_initializer=None,
                                   activation_fn=lrelu, reuse=reuse, scope='d_conv3', weights_initializer=initializer)
        # z: [32, 32]

        input = slim.convolution2d(input, 128, [3, 3], stride=[2, 2], padding="same", normalizer_fn=None,
                                  activation_fn=lrelu, reuse=reuse, scope='d_conv4', weights_initializer=initializer)
        # z: [16, 16]

        input = slim.convolution2d(input, 256, [3, 3], stride=[1, 1], padding="same", biases_initializer=None,
                                   activation_fn=lrelu, reuse=reuse, scope='d_conv5', weights_initializer=initializer)
        # z: [16, 16]

        input = slim.convolution2d(input, 256, [3, 3], stride=[2, 2], padding="same", normalizer_fn=None,
                                  activation_fn=lrelu, reuse=reuse, scope='d_conv6', weights_initializer=initializer)
        # z: [8, 8]

        input = slim.convolution2d(input, 512, [3, 3], stride=[1, 1], padding="same", biases_initializer=None,
                                   activation_fn=lrelu, reuse=reuse, scope='d_conv7', weights_initializer=initializer)
        # z: [8, 8]

        input = slim.convolution2d(input, 512, [3, 3], stride=[2, 2], padding="same", normalizer_fn=None,
                                  activation_fn=lrelu, reuse=reuse, scope='d_conv8', weights_initializer=initializer)
        # z: [4, 4]

        input = slim.fully_connected(slim.flatten(input), 1, activation_fn=None, reuse=reuse, scope='d_out',
                                     weights_initializer=initializer)

        print('**********************')
        print(input.get_shape())


    return input


def train():



    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim, random_dim], name='random_input')

    fake_image = generator(random_input, is_train=True, reuse=False)
    real_result = discriminator(real_image, reuse=False)
    fake_result = discriminator(fake_image, reuse=True)


    # Define loss function
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
    g_loss = -tf.reduce_mean(fake_result)

    # Add gradient penalty
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[-1]))
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp


    gp = gradient_penalty(real_image, fake_image, discriminator)
    d_loss += gp * gp_lambda


    # Add various tf summary variables
    g_summary = tf.summary.scalar('g_loss', g_loss)
    d_summary = tf.summary.scalar('d_loss', d_loss)
    #writer_d = tf.summary.FileWriter('./summaries/logs/plot_d_loss')
    #writer_g = tf.summary.FileWriter('./summaries/logs/plot_g_loss')
    writer = tf.summary.FileWriter('./summaries/cartoon_wgan_gp_4')
    writer.add_graph(tf.get_default_graph())
    #my_summary_op = tf.summary.merge_all()


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]

    # Define the optimizers for WGAN-GP
    #d_optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    #g_optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)

    #d_optim = tf.train.

    fake_sample = generator(random_input, is_train=False)

    start_time = time.time()

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    ckpt_dir = './checkpoints/cartoon_wgan_gp_4'
    mkdir(ckpt_dir + '/')
    load_checkpoint(ckpt_dir, sess)


    print('[*] start training...')

    fake_ipt_sample = np.random.uniform(-1.0, 1.0, size=[random_dim, random_dim, random_dim])
    #fake_ipt_sample = np.random.normal(size=[random_dim, random_dim, random_dim])


    d_iters = 5
    epoch = 200
    batch_epoch = len(data_pool) // (batch_size * d_iters)
    max_it = epoch * batch_epoch

    for it in range(max_it):
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        for k in range(d_iters):
            train_image = data_pool.batch()
            fake_image = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim, random_dim])
            #fake_image = np.random.normal(size=[batch_size, random_dim, random_dim])
            dLoss, _ = sess.run([d_summary, d_optim], feed_dict={random_input: fake_image, real_image: train_image})
        writer.add_summary(dLoss, it)

        fake_image = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim, random_dim])
        #fake_image = np.random.normal(size=[batch_size, random_dim, random_dim])
        gLoss, _ = sess.run([g_summary, g_optim], feed_dict={random_input: fake_image})
        writer.add_summary(gLoss, it)

        #summaries = sess.run(my_summary_op, feed_dict={random_input: fake_image, real_image: train_image})
        #writer.add_summary(summaries, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        '''

        if it % 20 == 0:
            #print('train:[%d],d_loss:%f,g_loss:%f' % (it, dLoss, gLoss), 'time: ', time.time() - start_time)
            summaries = sess.run(my_summary_op, feed_dict={random_input: fake_image, real_image: train_image})
            writer.add_summary(summaries, it)
        '''

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            fake_sample_opt = sess.run(fake_sample, feed_dict={random_input: fake_ipt_sample})
            save_dir = './sample_images_while_training/cartoon_wgan_gp_4'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(fake_sample_opt, 8, 8),
                          '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))


def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)



if __name__ == '__main__':

    train()
















