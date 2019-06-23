import numpy as np
import cv2
import random
import sys
import skimage
sys.path.append('../')

def data_augmentation(img):
    if random.random()>0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        img = cv2.erode(img, kernel)
    return img


def get_batch(items, height, width, train=True):
    x = []
    y = []
    for item in items:
        image_path = item.split(' ')[0]
        label = np.zeros(3755)
        label[int(item.split(' ')[-1].strip())] = 1
        img = cv2.imread(image_path, 0)
        if img is None:
            print(image_path)
            continue
        
        img = cv2.resize(img, (width, height))
        _, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
        if train:
            img = data_augmentation(img)

        img = img.reshape(width, height, 1)
        img = img / 255.0 - 0.5
        x.append(img)
        y.append(label)
    return x, y
        

def generator(path_file, batch_size, input_height, input_width, train=True):
    f = open(path_file, 'r')
    items = f.readlines()
    f.close()
    if train:
        while True:
            shuffled_items = []
            index = [n for n in range(len(items))]
            random.shuffle(index)
            for i in range(len(items)):
                shuffled_items.append(items[index[i]])
            for j in range(len(items) // batch_size):
                x, y = get_batch(shuffled_items[j * batch_size:(j + 1) * batch_size],
                                 input_height, input_width, train)
                yield np.array(x), np.array(y, dtype=np.int64)
    else:
        while True:
            for j in range(len(items) // batch_size):
                x, y = get_batch(items[j * batch_size:(j + 1) * batch_size],
                                 input_height, input_width, train)
                yield np.array(x), np.array(y, dtype=np.int64)

