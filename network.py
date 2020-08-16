import numpy as np
import paddle
import paddle.fluid as fluid


# 搭建网络
def generator(x):
    print('x', x.shape)
    # conv1
    conv1 = fluid.layers.conv2d(input=x,
                            num_filters=64,
                            filter_size=5,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv1',
                            data_format='NHWC')
    print('conv1', conv1.shape)
    conv1 = fluid.layers.batch_norm(conv1, momentum=0.99, epsilon=0.001)
    conv1 = fluid.layers.relu(conv1, name=None)
    # conv2
    conv2 = fluid.layers.conv2d(input=conv1,
                            num_filters=128,
                            filter_size=3,
                            dilation=1,
                            stride=2,
                            padding='SAME',
                            name='generator_conv2',
                            data_format='NHWC')
    print('conv2', conv2.shape)
    conv2 = fluid.layers.batch_norm(conv2, momentum=0.99, epsilon=0.001)
    conv2 = fluid.layers.relu(conv2, name=None)
    # conv3
    conv3 = fluid.layers.conv2d(input=conv2,
                            num_filters=128,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv3',
                            data_format='NHWC')
    print('conv3', conv3.shape)
    conv3 = fluid.layers.batch_norm(conv3, momentum=0.99, epsilon=0.001)
    conv3 = fluid.layers.relu(conv3, name=None)
    # conv4
    conv4 = fluid.layers.conv2d(input=conv3,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=2,
                            padding='SAME',
                            name='generator_conv4',
                            data_format='NHWC')
    print('conv4', conv4.shape)
    conv4 = fluid.layers.batch_norm(conv4, momentum=0.99, epsilon=0.001)
    conv4 = fluid.layers.relu(conv4, name=None)
    # conv5
    conv5 = fluid.layers.conv2d(input=conv4,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv5',
                            data_format='NHWC')
    print('conv5', conv5.shape)
    conv5 = fluid.layers.batch_norm(conv5, momentum=0.99, epsilon=0.001)
    conv5 = fluid.layers.relu(conv5, name=None)
    # conv6
    conv6 = fluid.layers.conv2d(input=conv5,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv6',
                            data_format='NHWC')
    print('conv6', conv6.shape)
    conv6 = fluid.layers.batch_norm(conv6, momentum=0.99, epsilon=0.001)
    conv6 = fluid.layers.relu(conv6, name=None)

    # 空洞卷积
    # dilated1
    dilated1 = fluid.layers.conv2d(input=conv6,
                            num_filters=256,
                            filter_size=3,
                            dilation=2,
                            padding='SAME',
                            name='generator_dilated1',
                            data_format='NHWC')
    print('dilated1', dilated1.shape)
    dilated1 = fluid.layers.batch_norm(dilated1, momentum=0.99, epsilon=0.001)
    dilated1 = fluid.layers.relu(dilated1, name=None)
    # dilated2
    dilated2 = fluid.layers.conv2d(input=dilated1,
                            num_filters=256,
                            filter_size=3,
                            dilation=4,
                            padding='SAME',
                            name='generator_dilated2',
                            data_format='NHWC') #stride=1
    print('dilated2', dilated2.shape)
    dilated2 = fluid.layers.batch_norm(dilated2, momentum=0.99, epsilon=0.001)
    dilated2 = fluid.layers.relu(dilated2, name=None)
    # dilated3
    dilated3 = fluid.layers.conv2d(input=dilated2,
                            num_filters=256,
                            filter_size=3,
                            dilation=8,
                            padding='SAME',
                            name='generator_dilated3',
                            data_format='NHWC')
    print('dilated3', dilated3.shape)
    dilated3 = fluid.layers.batch_norm(dilated3, momentum=0.99, epsilon=0.001)
    dilated3 = fluid.layers.relu(dilated3, name=None)
    # dilated4
    dilated4 = fluid.layers.conv2d(input=dilated3,
                            num_filters=256,
                            filter_size=3,
                            dilation=16,
                            padding='SAME',
                            name='generator_dilated4',
                            data_format='NHWC')
    print('dilated4', dilated4.shape)
    dilated4 = fluid.layers.batch_norm(dilated4, momentum=0.99, epsilon=0.001)
    dilated4 = fluid.layers.relu(dilated4, name=None)

    # conv7
    conv7 = fluid.layers.conv2d(input=dilated4,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            name='generator_conv7',
                            data_format='NHWC')
    print('conv7', conv7.shape)
    conv7 = fluid.layers.batch_norm(conv7, momentum=0.99, epsilon=0.001)
    conv7 = fluid.layers.relu(conv7, name=None)
    # conv8
    conv8 = fluid.layers.conv2d(input=conv7,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv8',
                            data_format='NHWC')
    print('conv8', conv8.shape)
    conv8 = fluid.layers.batch_norm(conv8, momentum=0.99, epsilon=0.001)
    conv8 = fluid.layers.relu(conv8, name=None)
    # deconv1
    deconv1 = fluid.layers.conv2d_transpose(input=conv8, 
                            num_filters=128, 
                            output_size=[64,64],
                            stride = 2,
                            name='generator_deconv1',
                            data_format='NHWC')
    print('deconv1', deconv1.shape)
    deconv1 = fluid.layers.batch_norm(deconv1, momentum=0.99, epsilon=0.001)
    deconv1 = fluid.layers.relu(deconv1, name=None)
    # conv9
    conv9 = fluid.layers.conv2d(input=deconv1,
                            num_filters=128,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv9',
                            data_format='NHWC')
    print('conv9', conv9.shape)
    conv9 = fluid.layers.batch_norm(conv9, momentum=0.99, epsilon=0.001)
    conv9 = fluid.layers.relu(conv9, name=None)
    # deconv2
    deconv2 = fluid.layers.conv2d_transpose(input=conv9, 
                            num_filters=64, 
                            output_size=[128,128],
                            stride = 2,
                            name='generator_deconv2',
                            data_format='NHWC')
    print('deconv2', deconv2.shape)
    deconv2 = fluid.layers.batch_norm(deconv2, momentum=0.99, epsilon=0.001)
    deconv2 = fluid.layers.relu(deconv2, name=None)
    # conv10
    conv10 = fluid.layers.conv2d(input=deconv2,
                            num_filters=32,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv10',
                            data_format='NHWC')
    print('conv10', conv10.shape)
    conv10 = fluid.layers.batch_norm(conv10, momentum=0.99, epsilon=0.001)
    conv10 = fluid.layers.relu(conv10, name=None)
    # conv11
    x = fluid.layers.conv2d(input=conv10,
                            num_filters=3,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv11',
                            data_format='NHWC')
    print('x', x.shape)
    x = fluid.layers.tanh(x)
    return x

def discriminator(global_x, local_x):
    def global_discriminator(x):
        # conv1
        conv1 = fluid.layers.conv2d(input=x,
                        num_filters=64,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv1',
                        data_format='NHWC')
        print('conv1', conv1.shape)
        conv1 = fluid.layers.batch_norm(conv1, momentum=0.99, epsilon=0.001)
        conv1 = fluid.layers.relu(conv1, name=None)
        # conv2
        conv2 = fluid.layers.conv2d(input=conv1,
                        num_filters=128,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv2',
                        data_format='NHWC')
        print('conv2', conv2.shape)
        conv2 = fluid.layers.batch_norm(conv2, momentum=0.99, epsilon=0.001)
        conv2 = fluid.layers.relu(conv2, name=None)
        # conv3
        conv3 = fluid.layers.conv2d(input=conv2,
                        num_filters=256,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv3',
                        data_format='NHWC')
        print('conv3', conv3.shape)
        conv3 = fluid.layers.batch_norm(conv3, momentum=0.99, epsilon=0.001)
        conv3 = fluid.layers.relu(conv3, name=None)
        # conv4
        conv4 = fluid.layers.conv2d(input=conv3,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv4',
                        data_format='NHWC')
        print('conv4', conv4.shape)
        conv4 = fluid.layers.batch_norm(conv4, momentum=0.99, epsilon=0.001)
        conv4 = fluid.layers.relu(conv4, name=None)
        # conv5
        conv5 = fluid.layers.conv2d(input=conv4,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv5',
                        data_format='NHWC')
        print('conv5', conv5.shape)
        conv5 = fluid.layers.batch_norm(conv5, momentum=0.99, epsilon=0.001)
        conv5 = fluid.layers.relu(conv5, name=None)
        # conv6
        conv6 = fluid.layers.conv2d(input=conv5,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv6',
                        data_format='NHWC')
        print('conv6', conv6.shape)
        conv6 = fluid.layers.batch_norm(conv6, momentum=0.99, epsilon=0.001)
        conv6 = fluid.layers.relu(conv6, name=None)
        # fc
        x = fluid.layers.fc(input=conv6, 
                        size=1024,
                        name='discriminator_global_fc1')
        return x

    def local_discriminator(x):
        # conv1
        conv1 = fluid.layers.conv2d(input=x,
                        num_filters=64,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_lobal_conv1',
                        data_format='NHWC')
        print('conv1', conv1.shape)
        conv1 = fluid.layers.batch_norm(conv1, momentum=0.99, epsilon=0.001)
        conv1 = fluid.layers.relu(conv1, name=None)
        # conv2
        conv2 = fluid.layers.conv2d(input=conv1,
                        num_filters=128,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_lobal_conv2',
                        data_format='NHWC')
        print('conv2', conv2.shape)
        conv2 = fluid.layers.batch_norm(conv2, momentum=0.99, epsilon=0.001)
        conv2 = fluid.layers.relu(conv2, name=None)
        # conv3
        conv3 = fluid.layers.conv2d(input=conv2,
                        num_filters=256,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_lobal_conv3',
                        data_format='NHWC')
        print('conv3', conv3.shape)
        conv3 = fluid.layers.batch_norm(conv3, momentum=0.99, epsilon=0.001)
        conv3 = fluid.layers.relu(conv3, name=None)
        # conv4
        conv4 = fluid.layers.conv2d(input=conv3,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_lobal_conv4',
                        data_format='NHWC')
        print('conv4', conv4.shape)
        conv4 = fluid.layers.batch_norm(conv4, momentum=0.99, epsilon=0.001)
        conv4 = fluid.layers.relu(conv4, name=None)
        # conv5
        conv5 = fluid.layers.conv2d(input=conv4,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_lobal_conv5',
                        data_format='NHWC')
        print('conv5', conv5.shape)
        conv5 = fluid.layers.batch_norm(conv5, momentum=0.99, epsilon=0.001)
        conv5 = fluid.layers.relu(conv5, name=None)
        # fc
        x = fluid.layers.fc(input=conv5, 
                        size=1024,
                        name='discriminator_lobal_fc1')
        return x

    global_output = global_discriminator(global_x)
    local_output = local_discriminator(local_x)
    print('global_output',global_output.shape)
    print('local_output',local_output.shape)
    output = fluid.layers.concat([global_output, local_output], axis=1)
    output = fluid.layers.fc(output, size=1,name='discriminator_concatenation_fc1')

    return output

# L2_loss
def L2_loss(yhat, y):
    loss = np.dot(y-yhat, y-yhat)
    loss.astype(np.float32)
    return loss

# 定义域损失函数
def calc_g_loss(x, completion):
    loss = L2_loss(x, completion)
    return fluid.layers.reduce_mean(loss)

def calc_d_loss(real, fake):
    alpha = 0.1
    d_loss_real = fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(x=real, label=fluid.layers.ones_like(real)))
    d_loss_fake = fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(x=fake, label=fluid.layers.zeros_like(fake)))
    return fluid.layers.elementwise_add(d_loss_real, d_loss_fake) * alpha
