from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.optimizers import SGD
import numpy as np
import cv2

import ImageLoader as il

sliding_window_size = 50
row_size = 250
scale_size = 600
btch_size = 50
epoch_amount = 10

images = il.load_resized_imgs("Data/wider_face_train_bbx_gt.txt", 600)
samples = []
labels = []


# -----utilities-----
def convert_to_trainingdata():
    for image in images:
        sfw = image.get("sfw", None)
        sfh = image.get("sfh", None)
        array = np.zeros(shape=(1,1,84681))

        positions = image.get("positions", None)

        if(positions != None):
            for position in positions:
                x = position.get("x", None)
                y = position.get("y", None)
                w = position.get("width", None)
                h = position.get("height", None)

                array[0,0,convert_coordinates(x, y, w, h, sfw, sfh)] = 1
        
        samples.append(image.get("img", None))
        labels.append(array)


def convert_coordinates(x, y, w, h, sfw, sfh):
    x = int(x * sfw)
    y = int(y * sfh)
    w = int(w * sfw)
    h = int(h * sfh)

    # needs to be tested
    if(w > 20):
        x = x + int((w - 20) / 2)

    if(w < 20):
        x = x - int((20 - w) / 2)
    
    if(h > 20):
        y = y + int((h - 20) / 2)

    if(h < 20):
        y = y - int((20 - h) / 2)
        
    if(x > 580):
        x = 580

    if(y > 580):
        y = 580

    # interim result 
    index_x = int(x/2)
    index_y = int(y/2)

    # index for 1d-np-array
    index_array = int (row_size * index_y) + index_x

    # print("x: ", x)
    # print("y: ", y)
    # print("index_x: ", index_x)
    # print("index_y: ", index_y)
    # print("index_array: ", index_array)
    
    return index_array


convert_to_trainingdata()

x_train = np.array(samples[:int(len(samples) / 2)])
x_test = np.array(samples[int(len(samples) / 2):])

del samples[:]

y_train = np.array(labels[:int(len(labels) / 2)])
y_test = np.array(labels[int(len(labels) / 2):])

del labels[:]


# -----model-----

model = Sequential([
    # width x hight x dimension
    # 600x600x3
    Conv2D(10, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform", input_shape=(600, 600, 3)),
    # 596x596x10
    MaxPool2D((2,2), data_format="channels_last"),
    # 298x298x10
    Conv2D(20, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 294x294x20
    MaxPool2D((2,2), data_format="channels_last"),
    # 147x147x20
    Conv2D(20, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 143x143x20
    MaxPool2D((2,2), data_format="channels_last"),
    # 71x71x20
    Conv2D(20, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 67x67x20
    MaxPool2D((3,3), data_format="channels_last"),
    # 22x22x20
    Conv2D(9680, 22, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 1x1x9680
    Conv2D(9680, 1, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 1x1x9680
    Conv2D(84681, 1, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform", activation="softmax")
    # 1x1x84681 (Output --> 291x291; each Pixel equals one Bounding Box with fix sized shape of 20x20)
])

model.summary()

model.compile(
    SGD(lr=0.001), 
    loss="mean_squared_error",  
    metrics=[
        "accuracy", 
        "binary_accuracy", 
        "categorical_accuracy"
    ]
)

model.fit(
    x_train,
    y_train,
    batch_size=btch_size,
    epochs=epoch_amount,
    verbose=2
)

score = model.evaluate(
    x_test,
    y_test,
    batch_size=btch_size,
    verbose=1
)