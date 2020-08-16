# PaddlePaddle_Image_Completion
本项目代码使用 PaddlePaddle 框架进行实现，应用场景：图像补全(Image completion)，目标移除(Object remove)

## 图像补全 -- Globally and Locally Consistent Image Completion
- #### 文章来源：SIGGRAPH  2017

- #### 下载链接：[Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/en/)


- #### 应用场景：图像补全(Image completion)，目标移除(Object remove)


- #### 使用的数据集是[CelebA人脸数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)，已下载放置在此项目的数据集中

- #### **目标**：
#### 进行图像填充，填充任意形状的缺失区域来完成任意分辨率的图像。

#### 在此篇论文中，作者们提出了一中图像补全方法，可以使得图像的缺失部分自动补全，局部和整图保持一致。作者通过全卷积网络，可以补全图片中任何形状的缺失，为了保持补全后的图像与原图的一致性，作者使用全局（整张图片）和局部（缺失补全部分）两种鉴别器来训练。全局鉴别器查看整个图像以评估它是否作为整体是连贯的，而局部鉴别器仅查看以完成区域为中心的小区域来确保所生成的补丁的局部一致性。 然后对图像补全网络训练以欺骗两个内容鉴别器网络，这要求它生成总体以及细节上与真实无法区分的图像。我们证明了我们的方法可以用来完成各种各样的场景。 此外，与PatchMatch等基于补丁的方法相比，我们的方法可以生成图像中未出现的碎片，这使我们能够自然地完成具有熟悉且高度特定的结构（如面部）的对象的图像。

- #### **网络构造**：
#### 完成网络：完成网络是完全卷积的，用来修复图像。
#### 全局上下文鉴别器：以完整的图像作为输入，识别场景的全局一致性。
#### 局部上下文鉴别器：只关注完成区域周围的一个小区域，以判断更详细的外观质量。
#### 对图像完成网络进行训练，以欺骗两个上下文鉴别器网络，这要求它生成在总体一致性和细节方面与真实图像无法区分的图像。

- #### **网络架构**:

#### 此网络由一个完成网络和两个辅助上下文鉴别器网络组成，这两个鉴别器网络只用于训练完成网络，在测试过程中不使用。全局鉴别器网络以整个图像为输入，而局部鉴别器网络仅以完成区域周围的一小块区域作为输入。训练两个鉴别器网络以确定图像是真实的还是由完成网络完成的，而生成网络被训练来欺骗两个鉴别器网络，使生成的图像达到真实图像的水平。
![image](https://img-blog.csdnimg.cn/20200816132702671.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70#pic_center)


- #### **补全网络结构**
#### 补全网络先利用卷积降低图片的分辨率然后利用去卷积增大图片的分辨率得到修复结果。为了保证生成区域尽量不模糊，文中降低分辨率的操作是使用strided convolution 的方式进行的，而且只用了两次，将图片的size 变为原来的四分之一。同时在中间层还使用了**空洞卷积**来增大感受野，在尽量获取更大范围内的图像信息的同时不损失额外的信息。

![image](https://img-blog.csdnimg.cn/20200816132309598.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70#pic_center) **空洞卷积**
![image](https://pic1.zhimg.com/50/v2-4959201e816888c6648f2e78cccfd253_hd.webp?source=1940ef5c)

- ### 内容鉴别器
#### 这些网络基于卷积神经网络，将图像压缩成小特征向量。 网络的输出通过连接层融合在一起，连接层预测出图像是真实的概率的一个连续值。网络结构如下：
![image](https://img-blog.csdnimg.cn/20200816132527575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70#pic_center)

#### 原文作者使用4个K80 GPU，使用的输入图像大小是256*256，训练了2个月才训练完成。
#### 本项目为了缩短训练时间，对训练方式做了简化，使用的输入图像大小：128*128，训练方式改为：先训练生成器再将生成器和判别器一起训练。

### **代码实现**

- #### 加载数据集
```
python to_npy.py
```
- #### 开始训练
```
python train.py
```
- #### 模型测试
```
python test.py
```
- #### 效果展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200816134144523.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNTQ5NjEy,size_16,color_FFFFFF,t_70#pic_center)
