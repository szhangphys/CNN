# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

import inspect
import time
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
plt.switch_backend('agg')

import os    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error     
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error  

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

def compute_accuracy(images_test, labels_test):#计算精度
    global prediction_
    tests_prediction = sess.run(prediction_, feed_dict={xs: images_test, keep_prob: 1})
    tests_prediction = np.array(tests_prediction)
    accuracy = roc_auc_score(labels_test,tests_prediction)
    FPR,TPR,THR=roc_curve(labels_test,tests_prediction)
    return accuracy,FPR,TPR,THR

def conv_add_layer(inputs,in_size,out_size,n_layer,activation_function=None,strides=[1, 1, 1, 1],padding='SAME',): #activation_function=None线性函数，卷积层    
    layer_name="conv_layer%s" % n_layer

    with tf.name_scope(layer_name):    
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.truncated_normal([3,3,in_size,out_size],mean=1e-5,stddev=0.5)) #Weight中都是随机变量    
            tf.summary.histogram(layer_name+"/weights",Weights) #可视化观看变量    
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)#tf.constant(0.1, shape=shape)) #biases推荐初始值不为0    
            tf.summary.histogram(layer_name+"/biases",biases) #可视化观看变量    
        with tf.name_scope('Wx_plus_b'):    
            Wx_plus_b = tf.nn.conv2d(inputs,Weights, strides=strides, padding=padding)+biases #inputs*Weight+biases    
            tf.summary.histogram(layer_name+"/Wx_plus_b",Wx_plus_b) #可视化观看变量    
        if activation_function is None:    
            outputs = Wx_plus_b    
        else:    
            outputs = activation_function(Wx_plus_b)    
            tf.summary.histogram(layer_name+"/outputs",outputs) #可视化观看变量   

        return outputs,Weights,biases    


def Batch_Norm(input,n_layer):#批归一化无/alpha /beta
    #layer_name="BN_%s" % n_layer
    batch_mean, batch_var = tf.nn.moments(input,[0])
    outputs=tf.nn.batch_normalization(input,mean=batch_mean,variance=batch_var,scale=None,offset=None,variance_epsilon=1e-10)
    return outputs

def Batch_Norm_(input,n_layer,out_size):#批归一化有/alpha /beta
    layer_name="BN_%s" % n_layer
    batch_mean, batch_var = tf.nn.moments(input,[0])
    gamma = tf.get_variable(layer_name+'/gamma',out_size,initializer=tf.constant_initializer(1))
    beta  = tf.get_variable(layer_name+'/beta', out_size,initializer=tf.constant_initializer(0))    
    outputs = tf.nn.batch_normalization(input,mean=batch_mean,variance=batch_var,scale=gamma,offset=beta,variance_epsilon=1e-10)   
    return outputs



def max_pool_2x2(x):#池化层
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def fc_layer(inputs,in_size,out_size,n_layer,activation_function=None):#全连接层
    layer_name="fc_%s" % n_layer
    with tf.name_scope(layer_name):    
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.truncated_normal([in_size,out_size],mean=1e-5,stddev=0.5)) #Weight中都是随机变量    
            tf.summary.histogram(layer_name+"/weights",Weights) #可视化观看变量    
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.constant(0.1, shape=[out_size]))#tf.constant(0.1, shape=shape)) #biases推荐初始值不为0    
            tf.summary.histogram(layer_name+"/biases",biases) #可视化观看变量   
        with tf.name_scope('Wx_plus_b'):    
            Wx_plus_b = tf.matmul(inputs,Weights)+biases #inputs*Weight+biases    
            tf.summary.histogram(layer_name+"/Wx_plus_b",Wx_plus_b) #可视化观看变量    
        if activation_function is None:    
            outputs = Wx_plus_b    
        else:    
            outputs = activation_function(Wx_plus_b)    
            tf.summary.histogram(layer_name+"/outputs",outputs) #可视化观看变量   
    return outputs,Weights,biases




def create_images(data, n_theta_bins, n_phi_bins, n_time_bins):   #数据中按事例统计，三维数组/图片
    images = []
    event_indexes = {}
    event_ids = np.unique(data['EventID'].values)    
    # collect event indexes
    data_event_ids = data['EventID'].values
    for i in range(len(data)):
        i_event = data_event_ids[i]
        if i_event in event_indexes:
            event_indexes[i_event].append(i)
        else:
            event_indexes[i_event] = [i]

    for i_event in event_ids:
        event = data.iloc[event_indexes[i_event]]
        X = event[['Theta', 'Phi', 'Time']].values
        one_image, edges = np.histogramdd(X, bins=(n_theta_bins, n_phi_bins, n_time_bins))
        images.append(one_image,)
    return np.array(images)

class Dataset(object):#数据集分类

    def __init__(self,images,labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = labels.shape[0]
        #self._num_examples 是指所有训练数据的样本个数
        pass
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
        
    def next_batch(self, batch_size,  shuffle=True):
    
        start = self._index_in_epoch  #self._index_in_epoch  所有的调用，总共用了多少个样本，相当于一个全局变量 #start第一个batch为0，剩下的就和self._index_in_epoch一样，如果超过了一个epoch，在下面还会重新赋值。
        # 第一个Batch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            index1 = np.arange(self._num_examples)  #生成的一个所有样本长度的np.array
            np.random.shuffle(index1)
           
            self._images = self.images[index1]
            self._labels = self.labels[index1]

           #从这里到下一个else，所做的是一个epoch快运行完了，但是不够一个batch，将这个epoch的结尾和下一个epoch的开头拼接起来，共同组成一个batch——size的数据。

        if start + batch_size > self._num_examples:
            # 1 Batch 结束
            self._epochs_completed += 1
            print('epoch completed:',self._epochs_completed)

            rest_num_examples = self._num_examples - start  # 一个epoch 最后不够一个batch还剩下几个
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
        # Shuffle 
            if shuffle:
                index2 = np.arange(self._num_examples)
                np.random.shuffle(index2)
                #print(self.labels[index2],index2)
                self._images = self.images[index2]
                self._labels = self.labels[index2]

              # 开始下一个batch
                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch
                #print(end)
                images_new_part = self._images[start:end] 
                labels_new_part = self._labels[start:end]
                return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
              #新的epoch，和上一个epoch的结尾凑成一个batch
        else:
            self._index_in_epoch += batch_size  #每调用这个函数一次，_index_in_epoch就加上一个batch——size的，它相当于一个全局变量，上不封顶
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]





#加载VGG16提取特征的输出
images1=np.load('photos/beta_data_56/codes_beta_pool5_2w.npy')
images2=np.load('photos/beta_data_56/codes_beta_pool5_4w.npy')
images3=np.load('photos/beta_data_56/codes_beta_pool5_6w.npy')
images4=np.load('photos/beta_data_56/codes_beta_pool5_8w.npy')


labels=np.load('photos/beta_data_224/labels(6w_32_64_3).npy')
images=np.concatenate((images1,images2,images3,images4))
#images= images[5000:35000]
#labels=labels[25000:55000]
(l,w,h)=(7,7,512)
labels_train = labels

print(labels_train.shape,images.shape)

#分割训练集，测试集，验证集
train_images, images_test_, train_labels, labels_test_ = train_test_split(images, labels_train, test_size=0.2)

val_images, images_test, val_labels, labels_test = train_test_split(images_test_,labels_test_, test_size=0.5)

#train_labels=np.reshape(train_labels['Label'].values,[-1,1])
#val_labels=np.reshape(val_labels0['Label'].values,[-1,1])
#labels_test=np.reshape(labels_test0['Label'].values,[-1,1])

print(train_labels.shape,type(train_labels))






#以上 ，总输入 数据，并选取训练集（images：(64000, 16, 24, 3)；labels：(64000, 1) ;验证集和测试集（images：(8000, 16, 24, 3)；labels： (8000, 1)）
with tf.name_scope('inputs'): #结构化    
    xs = tf.placeholder(tf.float32,[None,l,w,h],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')



#tensflow输入
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,l,w,h])
y_ = tf.reshape(ys,[-1,1])


#VGG16标准结构
'''

#x_image = Batch_Norm_(x_image,0,out_size=[28,28,256])

h_conv1_1=conv_add_layer(x_image,h,64,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv1_2=conv_add_layer(h_conv1_1,64,64,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_pool1 = max_pool_2x2(h_conv1_2)#112x112x64

h_conv2_1=conv_add_layer(h_pool1,64,128,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv2_2=conv_add_layer(h_conv2_1,128,128,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_pool2 = max_pool_2x2(h_conv2_2)#56x56x128

h_conv3_1=conv_add_layer(h_pool2,128,256,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv3_2=conv_add_layer(h_conv3_1,256,256,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv3_3=conv_add_layer(h_conv3_2,256,256,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_pool3 = max_pool_2x2(h_conv3_3)#28x28x256

h_conv4_1=conv_add_layer(h_pool3,h,512,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv4_2=conv_add_layer(h_conv4_1,512,512,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv4_3=conv_add_layer(h_conv4_2,512,512,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_pool4 = max_pool_2x2(h_conv4_3)#14x14x512


h_conv5_1=conv_add_layer(h_pool4,512,512,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv5_2=conv_add_layer(h_conv5_1,512,512,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_conv5_3=conv_add_layer(h_conv5_2,512,512,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_pool5 = max_pool_2x2(h_conv5_3)#7x7x512

'''





h_pool4_flat = tf.reshape(x_image, [-1, 7*7*512])
h_pool3 = Batch_Norm_(h_pool4_flat,3,out_size=[7*7*512])


h_fc1 =fc_layer(h_pool3,7*7*512,1024,6,activation_function=tf.nn.relu)[0]
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#h_fc1_drop = Batch_Norm(h_fc1_drop,6)
h_fc1_drop = Batch_Norm_(h_fc1_drop,6,out_size=[1024])


h_fc2 =fc_layer(h_fc1_drop,1024,64,7,activation_function=tf.nn.relu)[0]
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


#h_fc2_drop = Batch_Norm(h_fc2_drop,7)
h_fc2_drop = Batch_Norm_(h_fc2_drop,7,out_size=[64])

prediction = fc_layer(h_fc2_drop,64,1,8,activation_function=None)[0]
prediction_=tf.nn.sigmoid(prediction)



#交叉熵和loss
with tf.name_scope('loss'):    
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels=y_, logits=prediction, name=None))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('train'):  
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)







'''
#权重L2正则化
vars   = tf.trainable_variables() 
#print (vars)
#vars=vars[::3]
vars = [vars[2],vars[8],vars[12],vars[16]]#vars[14],vars[18],vars[22]]
print (vars)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels=y_, logits=prediction, name=None))

lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.005
with tf.name_scope('loss'):    
    loss = cross_entropy + lossL2
   # tf.summary.scalar('loss',loss)

with tf.name_scope('train'):  
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

'''


sess = tf.Session()

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess.run(init)
sess.run(tf.local_variables_initializer())


#合并到Summary中    
#merged = tf.summary.merge_all()    
#选定可视化存储目录    
#writer = tf.summary.FileWriter("Desktop/",sess.graph) 

train_loss_ = None

val_loss_ = None
saver = tf.train.Saver()


train_loss_ = None

val_loss_ = None

batch_data=Dataset(train_images,train_labels)
batch_data.__init__(train_images,train_labels)



for i in range(1750):

    train_images1,train_labels1=batch_data.next_batch( batch_size=256)
    train_images1,train_labels1
    ##print(train_images1.shape,type(train_images1))
    ##print(train_labels1.shape,type(train_labels1))
    sess.run(train_step, feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5})

    if i % 50 == 0:
        print('train loss:---------------------->>>>:',sess.run(cross_entropy,feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5}))


    if i % 250 == 0:
        print('test  loss:--------->>>>:',sess.run(cross_entropy,feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                                    ,'accuracy:',compute_accuracy(images_test,labels_test)[0] )
    
    if i % 250 == 0:

        train_loss = [sess.run(cross_entropy,feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5})]
        #    print(train_loss,type(train_loss))

        if train_loss_ is None:
            train_loss_ = train_loss

        else:
            train_loss_ = np.concatenate((train_loss_, train_loss),axis=0)
            #print(train_loss_)
        val_loss = [sess.run(cross_entropy,feed_dict={xs: val_images, ys: val_labels, keep_prob: 1})]

        if val_loss_ is None:
            val_loss_ = val_loss
        else:
            val_loss_ = np.concatenate((val_loss_, val_loss))




#epochs-loss图
plt.subplot(2,1,2)
plt.plot(compute_accuracy(images_test,labels_test)[1], compute_accuracy(images_test,labels_test)[2], label = 'auc = '+ str(compute_accuracy(images_test,labels_test)[0]) )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
#plt.show()
#plt.savefig("beta_try/pool5/roc_pool5.jpg")

#储存数据
np.save('beta_try_3/pool5/compute_accuracy_x',compute_accuracy(images_test,labels_test)[1])#ROC曲线FPR
np.save('beta_try_3/pool5/compute_accuracy_y',compute_accuracy(images_test,labels_test)[2])#ROC曲线TPR

np.save('beta_try_3/pool5/compute_accuracy_acc',compute_accuracy(images_test,labels_test)[0])#测试集和验证集LOSS

np.save('beta_try_3/pool5/train_loss_',train_loss_)
np.save('beta_try_3/pool5/val_loss_',val_loss_)


#储存参数信息
saver.save(sess, "beta_try_3/pool5/pool5.ckpt")

print('test  loss:--------->>>>:',sess.run(cross_entropy,feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                        ,'accuracy:',compute_accuracy(images_test,labels_test)[0] )

#在保存的参数下对测试集预测
test_proba = pd.DataFrame(labels_test,columns=['Label'])
#test_proba = pd.DataFrame(labels_test0['EventID'].values, columns=['EventID'])
#test_proba['Label'] = labels_test
test_proba_proba =  sess.run(prediction_, feed_dict={xs: images_test, keep_prob: 1})
test_proba['Proba'] = test_proba_proba
print(test_proba)
test_proba.to_csv('beta_try_3/pool5/pool5_submission.csv', index=False, float_format='%.6f')#储存预测结果

#loss图
plt.subplot(2,1,1)
plt.plot(range(0,7),train_loss_,label = "train loss")
plt.plot(range(0,7),val_loss_,label = " Validation loss")
plt.legend(loc='upper right')


plt.savefig("beta_try_3/pool5/val_train_pool5.jpg")
