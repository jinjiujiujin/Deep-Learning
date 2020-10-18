# 参考代码：https://github.com/tgpcai/digit_recognition

"""
Created on 2020.10.17 17.55

使用model来识别图片

@author:zyyz
"""
import tensorflow.compat.v1 as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def weight_variables(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return w
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b
def model():
    with tf.variable_scope('data'):    
        x = tf.placeholder(tf.float32, [None, 784])
    with tf.variable_scope("conv1"):
        w_conv1 = weight_variables([5,5,1,32])
        b_conv1 = bias_variables([32])
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1,1,1,1], padding="SAME")+b_conv1)
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("conv2"):
        w_conv2 = weight_variables([5,5,32,64])
        b_conv2 = bias_variables([64])
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")+b_conv2)
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("fc"):
        w_fc = weight_variables([7*7*64, 10])
        b_fc = bias_variables([10])
        
        x_fc_reshape = tf.reshape(x_pool2, [-1,7*7*64])
        y_predict =tf.matmul(x_fc_reshape, w_fc)+b_fc

    return x, y_predict

def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1],rgb[:,:,2]
        gray = 0.2989*r+0.5870*g+0.1140*b
        return gray

def load_image(file=r'D:\_workPlace\Book\AI\CV\HandwrittenDigitRecognition\test.png'):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im/255.0*2.0-1.0 # 归一化到【-1~1】之间
    im = im.reshape([784]) # 修改形状，使其和placeholder要求的结构相同
    return im


def conv_fc(model_path):
    img = load_image()

    x, y_predict=model() 
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, model_path)
        
        prediction = tf.argmax(y_predict, 1)
        predint = prediction.eval(feed_dict={x:[img]}, session=sess)
        print("识别结果:", predint)


if __name__=="__main__":
    tf.disable_eager_execution()
    conv_fc("Model_20201017/model.ckpt")       
