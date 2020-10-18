# 参考代码：https://github.com/tgpcai/digit_recognition

"""
Created on 2020.10.17 17.55

使用CNN(Convolutional Neural Networks)卷积神经网络实现手写数字识别

@author:zyyz
"""


# tensorflow的设计理念称之为计算流图。
# 在编写程序时，首先构筑整个系统的graph，代码不会直接生效。
# 然后在实际运行时，启动一个session，程序才会运行。
# 这样做的好处：避免反复切换底层程序实际运行的上下文，tensorflow会帮你优化整个系统的代码。
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variables(shape):
    # 标准差为0.1的正态分布
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return w

def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape)) # 定义tensor常量
    return b

def model():
    
    with tf.variable_scope('data'): 
        # placeholder()函数是在神经网络构建graph的时候在模型中的占位，在执行的时候再赋具体的值
        # 此时并没有把要输入的数据传入模型，它只会分配必要的内存
        # https://www.jianshu.com/p/a23cf9be601f    
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 卷积层1 卷积：5*5*1 32个filter strides=1
    with tf.variable_scope("conv1"):
        w_conv1 = weight_variables([5,5,1,32])
        b_conv1 = bias_variables([32])

        # 对x进行形状的改变[None, 784] ---> [None, 28, 28, 1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # padding='SAME'这种方式在处理边缘像素时会对输入的矩阵外层包裹n层0，以保证当卷积核的中心位于原图片边缘的像素点时原先空白的地方现在用0补上。 
        # ReLU激活 线性整流函数，通常指f(x)=max(0, x) [None, 28, 28, 1] -----> [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1,1,1,1], padding="SAME")+b_conv1)
        # 池化2*2，步长2，[None, 28, 28, 32] -----> [None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 卷积层2 卷积:5*5*32 64个filter strides=1
    with tf.variable_scope("conv2"):
        w_conv2 = weight_variables([5,5,32,64])
        b_conv2 = bias_variables([64])

        # ReLU激活 [None, 14, 14, 32] -----> [None, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")+b_conv2)
        

        # 池化2*2，步长2，[None, 14, 14, 64] -----> [None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 全连接层 [None, 7,7,64] -----> [None, 7*7*64]*[7*7*64, 10]+[10]=[None,10]
    with tf.variable_scope("fc"):
        w_fc = weight_variables([7*7*64, 10])
        b_fc = bias_variables([10])
        
        x_fc_reshape = tf.reshape(x_pool2, [-1,7*7*64])
        # 矩阵乘法
        y_predict =tf.matmul(x_fc_reshape, w_fc)+b_fc

    return x, y_true, y_predict

def conv_fc():
    # 获取数据
    mnist = input_data.read_data_sets(r'D:\_workPlace\Book\AI\CV\HandwrittenDigitRecognition\MNIST_DATA', one_hot=True)

    x, y_true, y_predict=model()

    # 交叉熵损失
    with tf.variable_scope("soft_cross"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    
    # 梯度下降求出最小损失。
    # 注意在深度学习中，或者网络层次比较复杂的情况下，学习率通常不能太高 
    with tf.variable_scope("optimizer"):
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    # 准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(2000):# 训练两千次，每次50个数据
            mnist_x, mnist_y = mnist.train.next_batch(50)
            
            sess.run(train_op, feed_dict = {x: mnist_x, y_true: mnist_y})

            print("训练第%d步，准确率为：%f" % (i, sess.run(accuracy,feed_dict = {x: mnist_x, y_true: mnist_y})))

        saver.save(sess, "Model_20201017/model.ckpt")

if __name__=="__main__":
    tf.disable_eager_execution() # tensorflow2的eager和placeholder冲突，所以disable eager
    conv_fc()       
