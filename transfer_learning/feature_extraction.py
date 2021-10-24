# -*- coding: UTF-8 -*-
import inspect
import os

import numpy as np
import tensorflow as tf
import time
import pandas as pd


import skimage
import skimage.io
import skimage.transform

# 均值预设
VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            #获取Vgg16所在的位置
            path = inspect.getfile(Vgg16)
            # print("######")
            # print(path)
            # print("######")
            # os.pardir代表目录的上一级
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print(path)
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):

        start_time = time.time()
        print("build model started")

        rgb_scaled = rgb * 255.0

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        # 对输入图片进行一步预处理
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")#224X224X64
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")#224X224X64
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')#112X112X64

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")#112X112X128
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")#112X112X128
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')#56X56X128

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")#56X56X256
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")#56X56X256
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")#56X56X256
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')#28X28X256

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")#28X28X512
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")#28X28X512
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")#28X28X512
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')#14X14X512

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")#14X14X512
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")#14X14X512
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")#14X14X512
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')#7X7X512

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)#1X4096

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)#1X1000

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))



    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")








data_dir = 'photos/56/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

# 计算batch的值
batch_size = 50
# 用codes_list来存储特征值
codes_list = []

labels = []

batch = []

codes = None


labels = pd.read_csv("../labels_train.csv")
event1=labels['EventID'].values
labels=labels['Label'].values

for k in [2,4,6,8]:
    m=0+(k-2)*10000
    n=k*10000
    event=event1[m:n]
    #labels=labels[0+(k-2)*10:k*10]

    with tf.Session() as sess:
    # 构建VGG16模型对象
        vgg = Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
        # 载入VGG16模型
            vgg.build(input_)
        #codes = sess.run(vgg.relu6, feed_dict= {input_: images})
        ii=0
        for ID in event:
            class_path = data_dir
            #print(ID)
            files= '%i' %ID +'.npy' 
            img=np.load(os.path.join(class_path, files,))
            img = skimage.transform.resize(img, (224, 224))

            batch.append(img.reshape((1, 224, 224, 3)))
            ii+=1
        
            # 如果图片数量到了batch_size则开始具体的运算
            if ii % batch_size == 0 or ii == len(event):
                images = np.concatenate(batch)

                feed_dict = {input_: images}
                


                # 计算特征值
                codes_batch = sess.run(vgg.pool2, feed_dict=feed_dict)
                

                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))
                # 清空数组准备下一个batch的计算
                batch = []
                print('{} images processed'.format(ii))
    

    codes=np.save('photos/beta_data_56/codes_beta_pool2_'+'%i' %k+'w.npy',codes)
    #labels=np.save('beta_data_3/labels_beta'+'%i.jpg' %k+'w.npy',labels)

