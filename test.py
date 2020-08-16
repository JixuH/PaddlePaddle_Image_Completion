import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.fluid as fluid

import tqdm

sys.path.append('..')

IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 64
PRETRAIN_EPOCH = 100

test_npy = './npy/x_test.npy'

def test():
    # 原始数据
    x = fluid.layers.data(name='x',shape=[IMAGE_SIZE, IMAGE_SIZE, 3],dtype='float32')
    # 指定填充 1为洞
    mask = fluid.layers.data(name='mask',shape=[IMAGE_SIZE, IMAGE_SIZE, 1],dtype='float32')
    # 全局生成图
    global_completion = fluid.layers.data(name='global_completion',shape=[IMAGE_SIZE, IMAGE_SIZE, 3],dtype='float32')
    # 局部生成图
    local_completion = fluid.layers.data(name='local_completion',shape=[LOCAL_SIZE, LOCAL_SIZE, 3],dtype='float32')
    # 局部原图
    local_x = fluid.layers.data(name='local_x',shape=[LOCAL_SIZE, LOCAL_SIZE, 3],dtype='float32')

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(fluid.default_startup_program())
    # 获取训练和测试程序
    test_program = fluid.default_main_program().clone(for_test=True)
    # 加载模型
    save_pretrain_model_path = 'models/'
    fluid.io.load_params(executor=exe, dirname=save_pretrain_model_path, main_program = dg_program)

    x_test = np.load(test_npy)
    np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])
    print (len(x_test))
    step_num = int(len(x_test) / BATCH_SIZE)

    cnt = 0
    for i in tqdm.tqdm(range(step_num)):
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        _, mask_batch = get_points()
        completion_test = exe.run(dg_program, 
                                feed={'x': x_batch, 
                                        'mask': mask_batch,},
                                fetch_list=[completion])[0]
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            masked = np.array(raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255 , dtype=np.uint8)
            img = completion_test[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            # print(masked.shape)
            # print(img.shape)
            # print(raw.shape)
            dst = './output_test/{}.jpg'.format("{0:06d}".format(cnt))
            output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst)

def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)

    return np.array(points), np.array(mask)
    

def output_image(images, dst):
    fig = plt.figure()
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()

if __name__ == '__main__':
    test()
