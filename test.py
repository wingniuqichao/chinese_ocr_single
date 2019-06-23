from utils.data_generator import generator

if __name__ == '__main__':
    train = generator('data/hcl_train_data.txt', 2, 96, 96, True)
    
    import matplotlib.pyplot as plt 
    import numpy as np
    x, y = train.__next__()
    print(y)
    plt.figure()
    plt.imshow(x[0, :, :, 0], cmap='gray')
    print(np.argmax(y[0]))
    plt.show()
    x, y = train.__next__()
    plt.figure()
    plt.imshow(x[0, :, :, 0], cmap='gray')
    print(np.argmax(y[0]))
    plt.show()
    x, y = train.__next__()
    plt.figure()
    plt.imshow(x[0, :, :, 0], cmap='gray')
    print(np.argmax(y[0]))
    plt.show()