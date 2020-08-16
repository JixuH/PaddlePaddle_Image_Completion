import os

import numpy as np


# 加载数据
def load(dir_='./npy'):
    x_train = np.load(os.path.join(dir_, 'x_train.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test.npy'))
    return x_train, x_test

if __name__ == '__main__':
    x_train, x_test = load()
    print(x_train.shape)
    print(x_test.shape)
