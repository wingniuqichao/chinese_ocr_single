from net.net_keras import net
from utils.data_generator import generator
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import config
# from keras.utils import plot_model

K.clear_session()


def plot_history(history, result_dir, prefix):
    '''
    将训练与验证的accuracy与loss画出来
    '''
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(prefix)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(prefix)))
    plt.close()


def save_history(history, result_dir, prefix):
    '''
    保存每一轮的结果
    '''
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

def step_decay(epoch):
    initial_lrate = 0.1

    initial_lrate = initial_lrate * 0.1**(epoch//10)
    if initial_lrate < 1.0e-5:
        initial_lrate = 1.0e-5
    return initial_lrate

def main():
    #-------------------参数设置------------------------
    train_size = np.loadtxt(config.TRAIN_FILE, dtype=str).shape[0]
    val_size = np.loadtxt(config.TEST_FILE, dtype=str).shape[0]

    model = net()

    # 设置变化的学习率
    # learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    learning_rate_reduction = LearningRateScheduler(step_decay)
    learning_rate = 0.1
    model.compile(loss='categorical_crossentropy',
                  # optimizer=Adam(lr=1e-4),
                  optimizer=SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    # plot_model(model, to_file='model.png', show_shapes=True)
    model.summary()
    # 生成图片读取迭代器，用多少读多少，不用一次性全部把图片读进内存，这样可以节省内存。
    train = generator(config.TRAIN_FILE, config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, True)
    val = generator(config.TEST_FILE, config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, False)

    if not os.path.exists('./results/'):
        os.mkdir('./results')
    if not os.path.exists('./weights/'):
        os.mkdir('./weights')
    # 每一轮保存历史最好模型，以验证集准确率为依据
    save_file = './weights/best_weights.h5'
    checkpoint = ModelCheckpoint(save_file, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
    history = model.fit_generator(train,
                        steps_per_epoch=train_size // config.BATCH_SIZE,
                        validation_data=val,
                        validation_steps=val_size // config.BATCH_SIZE,
                        epochs=config.EPOCHS,
                        callbacks=[checkpoint, learning_rate_reduction]
                                  )
    plot_history(history, './results/', 'Unet')
    save_history(history, './results/', 'Unet')


if __name__ == '__main__':
    main()