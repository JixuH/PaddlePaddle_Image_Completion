# to_npy 将图片读取完保存在.npy文件中

import glob
import os

import cv2
import numpy as np

ratio = 0.95
image_size = 128

x = []
paths = glob.glob('./img_align_celeba/*.jpg')
for path in paths[:20000]:
    # 读图
    img = cv2.imread(path)
    # resize
    img = cv2.resize(img, (image_size, image_size))
    # 改通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x.append(img)

x = np.array(x, dtype=np.float32)
np.random.shuffle(x)

p = int(ratio * len(x))
x_train = x[:p]
x_test = x[p:]
if not os.path.exists('./npy'):
    os.mkdir('./npy')
np.save('./npy/x_train.npy', x_train)
np.save('./npy/x_test.npy', x_test)
