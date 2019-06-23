
from keras.models import Model
from keras.layers import Input, Dropout, Reshape, Activation, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.initializers import orthogonal, constant, he_normal
from keras.regularizers import l2
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import config

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def net():
    inputs = Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS))
    x = Conv2D(config.FILTER_NUM[0], (1, 3), padding='same', kernel_initializer=he_normal())(inputs)
    x = Conv2D(config.FILTER_NUM[0], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Conv2D(config.FILTER_NUM[1], (1, 3), padding='same', kernel_initializer=he_normal())(x)
    x = Conv2D(config.FILTER_NUM[1], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Conv2D(config.FILTER_NUM[2], (1, 3), padding='same', kernel_initializer=he_normal())(x)
    x = Conv2D(config.FILTER_NUM[2], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Conv2D(config.FILTER_NUM[3], (1, 3), padding='same', kernel_initializer=he_normal())(x)
    x = Conv2D(config.FILTER_NUM[3], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = Conv2D(config.FILTER_NUM[4], (1, 3), padding='same', kernel_initializer=he_normal())(x)
    x = Conv2D(config.FILTER_NUM[4], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Conv2D(config.FILTER_NUM[5], (1, 3), padding='same', kernel_initializer=he_normal())(x)
    x = Conv2D(config.FILTER_NUM[5], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = Conv2D(config.FILTER_NUM[6], (1, 3), padding='same', kernel_initializer=he_normal())(x)
    x = Conv2D(config.FILTER_NUM[6], (3, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(config.FILTER_NUM[7], kernel_regularizer=l2(0.005), kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # x = Activation(relu6)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    y = Dense(config.NUM_LABELS, activation='softmax', kernel_initializer=he_normal())(x)
    model = Model(inputs=inputs, outputs=y)
    return model