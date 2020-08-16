import cv2
import numpy as np
import paddle
import paddle.fluid as fluid

import load
import tqdm
from network import *

# 参数定义
IMAGE_SIZE=128
LOCAL_SIZE=64
HOLE_MIN=24
HOLE_MAX=48
LEARNING_RATE=1e-3
BATCH_SIZE=64
use_gpu=True

d_program = fluid.Program()
dg_program = fluid.Program()

# 定义判别器的program
with fluid.program_guard(d_program):
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
    # 真实图fc
    real = discriminator(x, local_x)
    # 生成图fc
    fake = discriminator(global_completion, local_completion)

    # 计算生成图片被判别为真实样本的loss
    d_loss = calc_d_loss(real, fake)


# 定义判别生成图片的program
with fluid.program_guard(dg_program):
    # 原始数据
    x = fluid.layers.data(name='x',shape=[IMAGE_SIZE, IMAGE_SIZE, 3],dtype='float32')
    # 指定填充 1为洞
    mask = fluid.layers.data(name='mask',shape=[IMAGE_SIZE, IMAGE_SIZE, 1],dtype='float32')
    # 对原始数据挖空洞传入网络
    input_data = x * (1 - mask)
    #print('input_data',input_data)
    imitation = generator(input_data)
    # 修复完的图只保留空洞的部分和原图拼接
    completion = imitation * mask + x * (1 - mask)
    
    g_program = dg_program.clone()
    g_program_test = dg_program.clone(for_test=True)

    # 得到原图和修复图片的loss
    dg_loss = calc_g_loss(x, completion)
    print('g_loss_shape:',dg_loss.shape)

opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
opt.minimize(loss=d_loss)
parameters = [p.name for p in g_program.global_block().all_parameters()]
opt.minimize(loss=dg_loss, parameter_list=parameters)

# 数据集标准化
x_train, x_test = load.load()
#print (x_train.shape)
x_train = np.array([a / 127.5 - 1 for a in x_train])
#print (x_train[0])
x_test = np.array([a / 127.5 - 1 for a in x_test])

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program()) 

# 生成器优先迭代次数
NUM_TRAIN_TIMES_OF_DG = 100
# 总迭代轮次
epoch = 300

step_num = int(len(x_train) / BATCH_SIZE)

np.random.shuffle(x_train)


def train():
    for pass_id in range(epoch):
        # 训练生成器
        if pass_id <= NUM_TRAIN_TIMES_OF_DG:
            g_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_points()
                # print(x_batch.shape)
                # print(mask_batch.shape)
                dg_loss_n = exe.run(dg_program,
                                     feed={'x': x_batch, 
                                            'mask':mask_batch,},
                                     fetch_list=[dg_loss])[0]
                g_loss_value += dg_loss_n
            print('Pass_id:{}, Completion loss: {}'.format(pass_id, g_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            
            completion_n = exe.run(dg_program, 
                            feed={'x': x_batch, 
                                    'mask': mask_batch,},
                            fetch_list=[completion])[0][0]
            # 修复图片
            sample = np.array((completion_n + 1) * 127.5, dtype=np.uint8)
            # 原图
            x_im = np.array((x_batch[0] + 1) * 127.5, dtype=np.uint8)
            # 挖空洞输入图
            input_im_data = x_batch * (1 - mask_batch)
            input_im = np.array((input_im_data[0] + 1) * 127.5, dtype=np.uint8)
            output_im = np.concatenate((x_im,input_im,sample),axis=1)
            print(output_im.shape)
            cv2.imwrite('./output/pass_id:{}.jpg'.format(pass_id), cv2.cvtColor(output_im, cv2.COLOR_RGB2BGR))
            # 保存模型
            save_pretrain_model_path = 'models/'
            # 创建保持模型文件目录
            #os.makedirs(save_pretrain_model_path)
            fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path, main_program=dg_program)

        # 生成器判断器一起训练
        else:
            g_loss_value = 0
            d_loss_value = 0
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                points_batch, mask_batch = get_points()
                dg_loss_n = exe.run(dg_program,
                                     feed={'x': x_batch, 
                                            'mask':mask_batch,},
                                     fetch_list=[dg_loss])[0]
                g_loss_value += dg_loss_n

                completion_n = exe.run(dg_program, 
                                    feed={'x': x_batch, 
                                            'mask': mask_batch,},
                                    fetch_list=[completion])[0]
                local_x_batch = []
                local_completion_batch = []
                for i in range(BATCH_SIZE):
                    x1, y1, x2, y2 = points_batch[i]
                    local_x_batch.append(x_batch[i][y1:y2, x1:x2, :])
                    local_completion_batch.append(completion_n[i][y1:y2, x1:x2, :])
                local_x_batch = np.array(local_x_batch)
                local_completion_batch = np.array(local_completion_batch)
                d_loss_n  = exe.run(d_program,
                                    feed={'x': x_batch, 'mask': mask_batch, 'local_x': local_x_batch, 'global_completion': completion_n, 'local_completion': local_completion_batch},
                                    fetch_list=[d_loss])[0]
                d_loss_value += d_loss_n
            print('Pass_id:{}, Completion loss: {}'.format(pass_id, g_loss_value))
            print('Pass_id:{}, Discriminator loss: {}'.format(pass_id, d_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            completion_n = exe.run(dg_program, 
                            feed={'x': x_batch, 
                                    'mask': mask_batch,},
                            fetch_list=[completion])[0][0]
            # 修复图片
            sample = np.array((completion_n + 1) * 127.5, dtype=np.uint8)
            # 原图
            x_im = np.array((x_batch[0] + 1) * 127.5, dtype=np.uint8)
            # 挖空洞输入图
            input_im_data = x_batch * (1 - mask_batch)
            input_im = np.array((input_im_data[0] + 1) * 127.5, dtype=np.uint8)
            output_im = np.concatenate((x_im,input_im,sample),axis=1)
            print(output_im.shape)
            cv2.imwrite('./output/pass_id:{}.jpg'.format(pass_id), cv2.cvtColor(output_im, cv2.COLOR_RGB2BGR))
            # 保存模型
            save_pretrain_model_path = 'models/'
            # 创建保持模型文件目录
            #os.makedirs(save_pretrain_model_path)
            fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path, main_program = dg_program)

# 原图挖洞，构造mask
def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        # 构造local
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])
        # local中挖洞
        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        # 构造mask
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)
    return np.array(points), np.array(mask)

if __name__ == '__main__':
    train()
