# _*_ coding: utf-8 _*_
import os

import tensorflow as tf
import numpy as np
import net.net as net
import config
from utils.data_generator import generator

def train():
    x = tf.placeholder(tf.float32, [config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS], name='x_input')
    y_ = tf.placeholder(tf.int64, [None], name='y_input')
    is_training = tf.placeholder(tf.bool)

    regularizer = tf.contrib.layers.l2_regularizer(config.REGULARAZTION_RATE)

    y = net.net(x, is_training, regularizer)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.natural_exp_decay(config.LEARNING_RATE_BASE, config.EPOCHS*config.EVERY_EPOCHS, config.DECAY_STEP, config.DECAT_RATE, staircase=False)
    train_step = tf.train.AdamOptimizer(0.00001+learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        train_generator = generator(config.TRAIN_FILE, config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, True)
        test_generator = generator(config.TEST_FILE, config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, False)
        for i in range(config.EPOCHS):
            avg_loss = []
            avg_acc = []
            for j in range(config.EVERY_EPOCHS):
                xs, ys = train_generator.__next__()
                _, loss_value, acc = sess.run([train_step, loss, accuracy], feed_dict={x: xs, y_: ys, is_training: True})
                avg_loss.append(loss_value)
                avg_acc.append(acc)
                if j % 10 == 0:
                    percept = i / config.EVERY_EPOCHS
                    print("Epoch: %2d/%2d  iter: %5d/%5d "%(i, config.EPOCHS, j, config.EVERY_EPOCHS) + "["+"#"*int(30*percept) + "."*(30-int(30*percept)) + "] loss: %.4f acc: %.4f\r"%(np.mean(avg_loss), np.mean(avg_acc)), end='')
            
            test_avg_loss = []
            test_avg_acc = []
            for j in range(config.TEST_ITERS):
                xs, ys = test_generator.__next__()
                loss_value, acc = sess.run([loss, accuracy], feed_dict={x: xs, y_: ys, is_training: False})
                test_avg_loss.append(loss_value)
                test_avg_acc.append(acc)
            with open('log.txt', 'a+') as f:
                f.write("Epoch: %2d train_loss: %.4f train_acc: %.4f test_loss: %.4f test_acc: %.4f\n"%(i, np.mean(avg_loss), np.mean(avg_acc), np.mean(test_avg_loss), np.mean(test_avg_acc))) 

            
train()