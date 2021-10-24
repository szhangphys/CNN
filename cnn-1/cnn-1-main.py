import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import os    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error     
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error  

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve




def compute_accuracy(images_test, labels_test):#计算精度
    global prediction_
    tests_prediction = sess.run(prediction_, feed_dict={xs: images_test, keep_prob: 1})
    tests_prediction = np.array(tests_prediction)
    accuracy = roc_auc_score(labels_test,tests_prediction)
    FPR,TPR,THR=roc_curve(labels_test,tests_prediction)
    return accuracy,FPR,TPR,THR

def conv_add_layer(inputs,in_size,out_size,n_layer,activation_function=None,strides=[1, 1, 1, 1],padding='SAME',): #activation_function=None线性函数    
    layer_name="conv_layer%s" % n_layer
    with tf.name_scope(layer_name):    
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.truncated_normal([3,3,in_size,out_size],mean=1e-5,stddev=0.5)) #Weight中都是随机变量    
            tf.summary.histogram(layer_name+"/weights",Weights) #可视化观看变量    
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)#tf.constant(0.1, shape=shape))    
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

def Batch_Norm_(input,n_layer,out_size):#批归一化带/alpha /beta
    layer_name="BN_%s" % n_layer
    batch_mean, batch_var = tf.nn.moments(input,[0])
    gamma = tf.get_variable(layer_name+'/gamma',out_size,initializer=tf.constant_initializer(1))
    beta  = tf.get_variable(layer_name+'/beta', out_size,initializer=tf.constant_initializer(0))    
    outputs = tf.nn.batch_normalization(input,mean=batch_mean,variance=batch_var,scale=gamma,offset=beta,variance_epsilon=1e-10)   
    return outputs



def max_pool_2x2(x):#池化
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def fc_layer(inputs,in_size,out_size,n_layer,activation_function=None):#全连接
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




def create_images(data, n_theta_bins, n_phi_bins, n_time_bins):#数据中按事例统计，三维数组/图片   
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
        one_image, edges = np.histogramdd(X, bins=(n_theta_bins, n_phi_bins, n_time_bins))#以θ-φ为平面，对时间统计
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






#data read

'''
data_train = pd.read_csv('data_train.csv',dtype={'EventID':np.float32,'Theta': np.float32,'Phi':np.float32,'Time':np.float32,})
labels_train = pd.read_csv('labels_train.csv',dtype={'EventID':np.float32,'Label': np.float32,})
data_train.head()
labels_train.head()

#参数
#images:长 x 宽 x 高=l*w*h
#n_theta_bins=10
#n_phi_bins=20
#n_time_bins=6
(l,w,h)=(16,24,3)

images = create_images(data_train, 
                       n_theta_bins=l, 
                       n_phi_bins=w, 
                       n_time_bins=h)

labels=labels_train[['Label']].values
print(type(labels))
print(labels.shape)

print(type(images))
print(images.shape)
np.save('images(6w_16_24_3).npy',images)
#np.save('labels(6w_3).npy',labels)
'''
#储存和加载生成列表

#np.save('images(6w_10_20_6).npy',images)
#np.save('labels(6w_1).npy',labels)



images=np.load('F:\\Study\\Code\\Python\\Deeplearning\\beta\\data\\images(6w_16_24_3).npy')
#labels_=np.load('F:\\Study\\Code\\Python\\Deeplearning\\beta\\data\\labels(6w_16_24_3)_1.npy')
labels_train = pd.read_csv('F:\\Study\\Code\\Python\\Deeplearning\\beta\\labels_train.csv',dtype={'EventID':np.float32,'Label': np.float32,})
labels=labels_train['Label'].values
#labels= np.array(labels)

#print (type(images),type(labels),labels.shape)


#分割训练集，测试集，验证集
train_images, images_test_, train_labels, labels_test_ = train_test_split(images, labels_train, test_size=0.2)


val_images, images_test, val_labels0, labels_test0 = train_test_split(images_test_,labels_test_, test_size=0.5)

train_labels=np.reshape(train_labels['Label'].values,[-1,1])
val_labels=np.reshape(val_labels0['Label'].values,[-1,1])
labels_test=np.reshape(labels_test0['Label'].values,[-1,1])

print(train_labels.shape,type(train_labels))


print(val_labels[-1])
'''
val_labels = pd.DataFrame(val_labels,columns=['EventID','Label'])
val_labels=np.reshape(val_labels['Label'].values,[-1,1])
labels_test = pd.DataFrame(labels_test,columns=['EventID','Label'])
labels_test=np.reshape(labels_test['Label'].values,[-1,1])
'''

#定义统计Bin
(l,w,h)=(16,24,3)



print('images:', train_images.shape, images_test.shape,type(train_images))
print('labels:', train_labels.shape, labels_test.shape)


#以上 ，总输入 数据，并选取训练集（images：(64000, 16, 24, 3)；labels：(64000, 1) ;验证集和测试集（images：(8000, 16, 24, 3)；labels： (8000, 1)）
with tf.name_scope('inputs'): #结构化    
    xs = tf.placeholder(tf.float32,[None,l,w,h],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')


keep_prob = tf.placeholder(tf.float32)



x_image = tf.reshape(xs,[-1,l,w,h])

y_ = tf.reshape(ys,[-1,1])


#卷积层：5，BN层：5个 ，池化层：3个
#x_image = Batch_Norm(x_image,0)
x_image = Batch_Norm_(x_image,0,out_size=[16,24,3])


h_conv1=conv_add_layer(x_image,h,16,1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 

h_conv1 = max_pool_2x2(h_conv1) 

h_conv1 = Batch_Norm_(h_conv1,1_1,out_size=[8,12,16])



h_conv1=conv_add_layer(h_conv1,16,16,1_1,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0] 
h_pool1 = max_pool_2x2(h_conv1) 

#h_pool1 = Batch_Norm(h_pool1,1)
h_pool1 = Batch_Norm_(h_pool1,1,out_size=[4,6,16])



h_conv2=conv_add_layer(h_pool1,16,32,2,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0]    

#h_pool2 = Batch_Norm(h_pool2,2)
h_pool2 = Batch_Norm_(h_conv2,2,out_size=[4,6,32])



h_pool3=conv_add_layer(h_pool2,32,32,3,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0]
h_pool3 = max_pool_2x2(h_pool3)

#h_pool3 = Batch_Norm(h_pool3,3)
h_pool3 = Batch_Norm_(h_pool3,3,out_size=[2,3,32])



h_conv4=conv_add_layer(h_pool3,32,128,4,activation_function=tf.nn.relu,strides=[1, 1, 1, 1],padding='SAME')[0]   


#flat
h_pool4_flat = tf.reshape(h_conv4, [-1, 3*4*64])
h_pool5 = Batch_Norm_(h_pool4_flat,5,out_size=[3*4*64])


#全连接层：3，Drop：2个，BN层：2个
h_fc1 =fc_layer(h_pool5,384*2,64,6,activation_function=tf.nn.relu)[0]
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#h_fc1_drop = Batch_Norm(h_fc1_drop,6)
h_fc1_drop = Batch_Norm_(h_fc1_drop,6,out_size=[64])


h_fc2 =fc_layer(h_fc1_drop,64,16,7,activation_function=tf.nn.relu)[0]
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


#h_fc2_drop = Batch_Norm(h_fc2_drop,7)
h_fc2_drop = Batch_Norm_(h_fc2_drop,7,out_size=[16])

prediction = fc_layer(h_fc2_drop,16,1,8,activation_function=None)[0]
prediction_=tf.nn.sigmoid(prediction)


'''
#normal loss
with tf.name_scope('loss'):    
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels=y_, logits=prediction, name=None))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('train'):  
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
'''

#权重L2惩罚
vars   = tf.trainable_variables() 
#print (vars)
#vars=vars[::3]
vars = [vars[2],vars[6],vars[10],vars[14],vars[18],vars[22],vars[26],vars[30]]
#print (vars)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( labels=y_, logits=prediction, name=None))

lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * 0.005
with tf.name_scope('loss'):    
    loss = cross_entropy + lossL2
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):  
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)




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
merged = tf.summary.merge_all()    
#选定可视化存储目录    
#writer = tf.summary.FileWriter("Desktop/",sess.graph) 

train_loss_ = None

val_loss_ = None
saver = tf.train.Saver()
batch_data=Dataset(train_images,train_labels)
batch_data.__init__(train_images,train_labels)

for i in range(4500):

    train_images1,train_labels1=batch_data.next_batch( batch_size=256)
    #print(train_images1.shape)

    sess.run(train_step, feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5})
    if i % 10 == 0:
        print('train loss:---------------------->>>>:',sess.run(cross_entropy,feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5}))
        
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

        #print(train_loss.shape)
        #train_loss = np.array(train_loss)
        #train_loss_ = np.concatenate((train_loss_,train_loss))

        #train_loss.append(train_loss)
        #val_loss = sess.run(cross_entropy,feed_dict={xs: val_images, ys: val_labels, keep_prob: 0.5})
        #val_loss.append(val_loss)

        #result = sess.run(merged,feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5}) #merged也是需要run的    
        #writer.add_summary(result,i)

    #if i % 20 == 0:
        #print('train loss:---------------------->>>>:',sess.run(cross_entropy,feed_dict={xs: train_images1, ys: train_labels1, keep_prob: 0.5}))


    if i % 150 == 0:
        print('val   loss:--------->>>>:',sess.run(cross_entropy,feed_dict={xs: val_images, ys: val_labels, keep_prob: 1})
                                    ,'accuracy:',compute_accuracy(val_images,val_labels)[0] )


        print('test  loss:--------->>>>:',sess.run(cross_entropy,feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                                    ,'accuracy:',compute_accuracy(images_test,labels_test)[0] )


        #tensorboard
        #result = sess.run(merged,feed_dict={xs:x_data,ys:y_data}) #merged也是需要run的    
        #writer.add_summary(result,i)


#储存参数信息
saver.save(sess, "beta_try/16_24_3.ckpt")

print('test  loss:--------->>>>:',sess.run(cross_entropy,feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                        ,'accuracy:',compute_accuracy(images_test,labels_test)[0] )


#预测概率
test_proba = pd.DataFrame(labels_test,columns=['Label'])
#test_proba = pd.DataFrame(labels_test0['EventID'].values, columns=['EventID'])
#test_proba['Label'] = labels_test
test_proba_proba =  sess.run(prediction_, feed_dict={xs: images_test, keep_prob: 1})
test_proba['Proba'] = test_proba_proba 
print(test_proba)
test_proba.to_csv('submission.csv', index=False, float_format='%.6f')


#epochs-loss图
#plt.subplot(2,1,1)
plt.plot(range(0,18),train_loss_,label = "train loss")
plt.plot(range(0,18),val_loss_,label = " Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

#auc曲线
plt.plot(compute_accuracy(images_test,labels_test)[1], compute_accuracy(images_test,labels_test)[2], label = 'auc = '+ str(compute_accuracy(images_test,labels_test)[0]) )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()




#读取
'''
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('beta_try/16_24_3.ckpt'))
    
    print('test  loss:--------->>>>:',sess.run(cross_entropy,feed_dict={xs: images_test, ys: labels_test, keep_prob: 1})
                            ,'accuracy:',compute_accuracy(images_test,labels_test)[0] )


'''