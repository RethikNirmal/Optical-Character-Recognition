import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import cv2
import pandas as pd
from data_extract import *
from visualisation import *
from keras.utils import np_utils
data = read_process_h5("train/digitStruct.mat")
path = "train/"
max_digits = 7
image_size = (54,128)
checkpoint_path = '../checkpoints/model.hdf5'
resume_training = True
print("Data Read")
#visualize(data)
# print(data)
#data = generateData(data, 1000)
image = []
image_index = []
y_count = []
y_label = []
for i in data:
    img = cv2.imread(path + i['filename'])
    img = cv2.resize(img,(128,128))
    img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for j in range(0,i['length']):
        image.append(img)
        image_index.append(j)
        y_count.append(i['length'])
        if(i['labels'][j] == 10):
            y_label.append(0)
        else:
            y_label.append(i['labels'][j])
        
print(np.unique(y_label)) 
#print(y_label.unique())

Xidx = np_utils.to_categorical(image_index, max_digits)
ycount = np_utils.to_categorical(y_count, max_digits)
ylabel = np_utils.to_categorical(y_label, 10)

df = {'image': image,'image_index':Xidx,'y_count':ycount,'y_label':ylabel }
pd.DataFrame(list(zip(image, Xidx, ycount,ylabel)),
              columns=['Image','img_index', 'ycount','ylabel'])
df.to_csv("SVHN_train_data.csv",encoding = 'utf-8',index = False)
        

    