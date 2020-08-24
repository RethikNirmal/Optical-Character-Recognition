from keras.layers import Input, Dense, Flatten, Dropout, Concatenate
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import keras
from keras import callbacks
import cv2
from keras.utils import np_utils
import h5py
from data_extract import *
from sklearn.model_selection import train_test_split
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          featurewise_center = True,
          featurewise_std_normalization = True,
              width_shift_range=0.2,
              height_shift_range=0.15,
              shear_range=0.4
    )
max_digits = 7
image_size = (128,128)


#Vision model (Base model)
vision_model = Sequential()
vision_model.add(Conv2D(64,(5,5),padding = 'valid',activation = 'relu',input_shape = (128,128,1)))
vision_model.add(Conv2D(64,(3,3),padding = 'valid',activation = 'relu'))
vision_model.add(MaxPooling2D((3,3)))
vision_model.add(Dropout(0.15))
vision_model.add(BatchNormalization())
vision_model.add(Conv2D(128,(5,5),padding = 'valid',activation = 'relu'))
vision_model.add(Conv2D(128,(3,3),padding = 'valid',activation = 'relu'))
vision_model.add(MaxPooling2D((3,3)))
vision_model.add(Dropout(0.15))
vision_model.add(BatchNormalization())
vision_model.add(Conv2D(64,(5,5),activation = 'relu'))
vision_model.add(Conv2D(64,(3,3),activation = 'relu'))
vision_model.add(MaxPooling2D((3,3)))
vision_model.add(BatchNormalization())
vision_model.add(Flatten())
vision_model.add(Dense(1024,activation = 'relu'))
vision_model.add(Dropout(0.15))
vision_model.add(BatchNormalization())
vision_model.add(Dense(1024,activation = 'relu'))
# print(vision_model.summary())

#Counter model
counter_model = Sequential()
counter_model.add(Dense(512,activation = 'relu',input_shape = (1024,)))
counter_model.add(Dense(256,activation = 'relu'))
counter_model.add(Dropout(0.15))
counter_model.add(BatchNormalization())
counter_model.add(Dense(max_digits,activation = 'softmax'))
# print(counter_model.summary())

# define detector model
h_in_detector = Input(shape = (1024,))

idx_in_detector = Input(shape=(max_digits,))
yl = Concatenate()([h_in_detector, idx_in_detector]) 

yl = Dense(512, activation='relu')(yl)
yl = BatchNormalization()(yl)
yl = Dense(512, activation='relu')(yl)
yl = BatchNormalization()(yl)
yl = Dropout(0.2)(yl)
yl = Dense(10, activation='softmax')(yl)
detector_model = Model(input=[h_in_detector, idx_in_detector], output=yl, name='detector')
# print(detector_model.summary())
# print("Model build complete")
max_digits = 7
data = read_process_h5("train/digitStruct.mat")
path = "train/"

image_size = (128,128)
checkpoint_path = 'model.hdf5'
resume_training = True
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
image = np.array(image,ndmin = 4)
image = np.reshape(image,(73257,128,128,1))            
datagen.fit(image)
Xidx = np_utils.to_categorical(image_index, max_digits)
ycount = np_utils.to_categorical(y_count, max_digits)
ylabel = np_utils.to_categorical(y_label, 10)



# print("image shape:",image.shape)
# print("Y count shape",ycount.shape)
# print("X idx shape",Xidx.shape)
# print("Y label shape",ylabel.shape)
Ximg_in = Input(shape=(128, 128,1), name='train_input_img')
Xidx_in = Input(shape=(max_digits,), name='train_input_idx')

h = vision_model(Ximg_in)
# print("Vision model output:",h.shape)
yc = counter_model(h)
# print("Counter model output:",yc.shape)
yl = detector_model([h, Xidx_in])


train_graph = Model([Ximg_in, Xidx_in],[yc, yl])
train_graph.compile(optimizer='adagrad', loss=['categorical_crossentropy','categorical_crossentropy'], metrics=['accuracy'])
#train_graph.summary()
# define checkpoint callback


#image, Ximg_val, Xidx, Xidx_val, ycount, ycount_val, ylabel, ylabel_val = train_test_split(image, Xidx, ycount, ylabel, test_size=0.05)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
history = train_graph.fit(datagen.flow(image, Xidx, ycount, ylabel),
                          epochs=50
                          )

# model_json = model.to_json()
# with open("model.json","w") as file:
#     file.write(model_json)
    
# model.save_weights("Model.h5")
