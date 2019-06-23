# _*_ coding: utf-8 _*_
import tensorflow as tf
from config import*



def batch_norm_layer(value,is_training=False,name='batch_norm'):
    '''
    批量归一化  返回批量归一化的结果
    
    args:
        value:代表输入，第一个维度为batch_size
        is_training:当它为True，代表是训练过程，这时会不断更新样本集的均值与方差。当测试时，要设置成False，这样就会使用训练样本集的均值和方差。
              默认测试模式
        name：名称。
    '''
    if is_training is True:
        #训练模式 使用指数加权函数不断更新均值和方差
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #测试模式 不更新均值和方差，直接使用
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)

def net(input_tensor, is_training, regularizer):

    # 96x96x1 --> 96x96x96
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [3, 3, NUM_CHANNELS, FILTER_NUM[0]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [FILTER_NUM[0]], 
            initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn1 = batch_norm_layer(tf.nn.bias_add(conv1, conv1_biases), is_training)
        relu1 = tf.nn.relu(bn1)
    # 96x96x96 --> 48x48x96
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 48x48x96 --> 48x48x128
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [3, 3, FILTER_NUM[0], FILTER_NUM[1]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [FILTER_NUM[1]], 
            initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn2 = batch_norm_layer(tf.nn.bias_add(conv2, conv2_biases), is_training)
        relu2 = tf.nn.relu(bn2)
    # 48x48x128 --> 24x24x128
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 24x24x128 --> 24x24x160
    with tf.variable_scope('layer5-conv3'):
        conv3_weights = tf.get_variable('weight', [3, 3, FILTER_NUM[1], FILTER_NUM[2]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [FILTER_NUM[2]], 
            initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn3 = batch_norm_layer(tf.nn.bias_add(conv3, conv3_biases), is_training)
        relu3 = tf.nn.relu(bn3)
    # 24x24x160 --> 12x12x160
    with tf.variable_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 12x12x160 --> 12x12x256
    with tf.variable_scope('layer7-conv4'):
        conv4_weights = tf.get_variable('weight', [3, 3, FILTER_NUM[2], FILTER_NUM[3]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias', [FILTER_NUM[3]], 
            initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn4 = batch_norm_layer(tf.nn.bias_add(conv4, conv4_biases), is_training)
        relu4 = tf.nn.relu(bn4)
    # 12x12x256 --> 12x12x256
    with tf.variable_scope('layer8-conv5'):
        conv5_weights = tf.get_variable('weight', [3, 3, FILTER_NUM[3], FILTER_NUM[4]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable('bias', [FILTER_NUM[4]], 
            initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn5 = batch_norm_layer(tf.nn.bias_add(conv5, conv5_biases), is_training)
        relu5 = tf.nn.relu(bn5)
    # 12x12x256 --> 6x6x256
    with tf.variable_scope('layer9-pool5'):
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 6x6x256 --> 6x6x384
    with tf.variable_scope('layer10-conv6'):
        conv6_weights = tf.get_variable('weight', [3, 3, FILTER_NUM[4], FILTER_NUM[5]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable('bias', [FILTER_NUM[5]], 
            initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(pool5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn6 = batch_norm_layer(tf.nn.bias_add(conv6, conv6_biases), is_training)
        relu6 = tf.nn.relu(bn6)
    # 6x6x256 --> 6x6x384
    with tf.variable_scope('layer11-conv7'):
        conv7_weights = tf.get_variable('weight', [3, 3, FILTER_NUM[5], FILTER_NUM[6]], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv7_biases = tf.get_variable('bias', [FILTER_NUM[6]], 
            initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(relu6, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn7 = batch_norm_layer(tf.nn.bias_add(conv7, conv7_biases), is_training)
        relu7 = tf.nn.relu(bn7)
    # 6x6x256 --> 3x3x384
    with tf.variable_scope('layer12-pool7'):
        pool7 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FCN
    pool7_shape = pool7.get_shape().as_list()
    nodes = pool7_shape[1] * pool7_shape[2] * pool7_shape[3]

    reshaped = tf.reshape(pool7, [pool7_shape[0], nodes])

    with tf.variable_scope('layer13-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FILTER_NUM[7]],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FILTER_NUM[7]],
            initializer=tf.constant_initializer(0.0))
        bn8 = batch_norm_layer(tf.matmul(reshaped, fc1_weights) + fc1_biases, is_training)
        fc1 = tf.nn.relu(bn8)
        if is_training is True:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer14-fc2'):
        fc2_weights = tf.get_variable('weight', [FILTER_NUM[7], NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS],
            initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    
    return logit