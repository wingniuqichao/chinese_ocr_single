from net.net_keras import net
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import config
import cv2

width = 96
height = 96

data = np.loadtxt("dict_train.txt", dtype='str', delimiter=',')
labels = np.array(['' for _ in range(3755)])
for i, j in data:
    labels[int(i)] = j.strip()



model = net()
model.load_weights("./weights/best_weights.h5")


img_list = ['test/%d.png'%i for i in range(21, 27)]

x = []
images = []
for i in range(len(img_list)):

    img = cv2.imread(img_list[i], 1)
    images.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)

    # up, down, left, right = 0,0,0,0
    # for row in range(img.shape[0]):
    #     if img[row, :].sum()>0:
    #         up = row
    #         break 
    # for row in range(img.shape[0]-1, 0, -1):
    #     if img[row, :].sum()>0:
    #         down = row 
    #         break 
    # for col in range(img.shape[1]):
    #     if img[:, col].sum() > 0:
    #         left = col 
    #         break 
    # for col in range(img.shape[1]-1, 0, -1):
    #     if img[:, col].sum() > 0:
    #         right = col 
    #         break 
    # img = img[up:down, left:right]

    img = cv2.resize(img, (width, height))
    img = img.reshape(width, height, 1)
    x.append(img)

x = np.array(x)
x = x / 255.0 - 0.5

y = model.predict(x)

print("\n\n识别结果：")
print(labels[np.argmax(y, axis=1)])
print('\n\n')