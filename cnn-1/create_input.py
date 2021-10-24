import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

import os    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error     
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error  

import cv2

import tensorflow as tf



def create_images(data, n_theta_bins, n_phi_bins, n_time_bins):
    
    images = []
    event_indexes = {}
    event_ids = np.unique(data['EventID'].values)    

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
#        print(type(X),X.shape)
        one_image, edges = np.histogramdd(X, bins=(n_theta_bins, n_phi_bins, n_time_bins))
        images.append(one_image,)
    return np.array(images)


#data read

data_train = pd.read_csv('F:\\Study\\Code\\Python\\Deeplearning\\beta\\data_train.csv',dtype={'EventID':np.float32,'Theta': np.float32,'Phi':np.float32,'Time':np.float32,})
labels_train = pd.read_csv('F:\\Study\\Code\\Python\\Deeplearning\\beta\\labels_train.csv',dtype={'EventID':np.float32,'Label': np.float32,})
data_train.head()
labels_train.head()


print(type(data_train))

labels=labels_train[['EventID','Label']].values


#x=data_train[['Theta']].values
#y=data_train[['Phi']].values
#z=data_train[['Time']].values

#print(x.shape)
#print(type(x))

#参数
#images:长 x 宽 x 高=l*w*h
#n_theta_bins=10
#n_phi_bins=20
#n_time_bins=6

#空间坐标：θ-φ-T ------> 直角坐标：x-y-z
'''
theta=data_train[['Theta']].values
phi=data_train[['Phi']].values
r=data_train[['Time']].values

print(type(r),r)

x= 3*r *np.sin(theta)*np.cos(phi)
y= 3*r *np.sin(theta)*np.sin(phi)
z= 3*r * np.cos(theta)

print(type(z),z)

data_train.insert(4, 'x', x)
data_train.insert(5, 'y', y)
data_train.insert(6, 'z', z)
print(data_train)
'''

(l,w,h)=(16,24,3)
images = create_images(data_train, 
	                       n_theta_bins=l, 
	                       n_phi_bins=w, 
	                       n_time_bins=h)



np.save('F:\\Study\\Code\\Python\\Deeplearning\\beta\\data\\images(6w_16_24_3).npy', images)
np.save('F:\\Study\\Code\\Python\\Deeplearning\\beta\\data\\labels(6w_16_24_3)_1.npy',labels)

#储存和加载生成列表

#np.save('images(6w_10_20_6).npy',images)
#np.save('labels(6w_1).npy',labels)

'''
images=np.load('images(6w_60_30_3).npy')
labels=np.load('labels(6w_60_30_3).npy')


img = images[1][:, :, 1]
plt.imshow(img)
plt.colorbar()
plt.show()
'''
