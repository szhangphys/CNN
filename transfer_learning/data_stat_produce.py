import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
#matplotlib.use('Agg')

import os    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息    
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error     
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error  


from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt 


data_dir = '56/'



data_train = pd.read_csv('../../data_train.csv',dtype={'EventID':np.float32,'Theta': np.float32,'Phi':np.float32,'Time':np.float32,})
data=data_train
print(data.shape)
images = []
event_indexes = {}
event_ids = np.unique(data['EventID'].values)


# 统计
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
    #print(type(X),X.shape)
    one_image, edges = np.histogramdd(X, bins=(56, 56, 3))
    #print(one_image.shape,type(one_image))
    #one_image=np.array(one_image)
    #plt.imshow(one_image[:,:,0])
    #plt.show()
    #cv2.imwrite(data_dir+'%i.jpg' %i_event, one_image)
    #储存图片格式
    #scipy.misc.toimage(one_image).save(data_dir+'%i.jpg' %i_event)
    #储存数组
    np.save(data_dir+'%i' %i_event,one_image)
 