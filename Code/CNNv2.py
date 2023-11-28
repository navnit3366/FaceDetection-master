from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.optimizers import SGD
from keras.activations import elu
import numpy as np
import cv2

import ImageLoader as il


scale_size = 600

sliding_window_size = 20 # windows don't overlapp
row_size = int(scale_size / sliding_window_size) # 30

btch_size = 50
epoch_amount = 140

images = il.load_cropped_imgs("Data/wider_face_train_bbx_gt.txt", 600)
samples = []
labels = []

# -----utilities-----
def convert_coordinate_to_label(x, y, w, h, sfw, sfh):
    x = int(x * sfw)
    y = int(y * sfh)
    w = int(w * sfw)
    h = int(h * sfh)

    x_realtive = x % sliding_window_size
    y_relative = y % sliding_window_size

    index_x = int(x / sliding_window_size)
    index_y = int(y / sliding_window_size)

    # print("x_relative: ", x_realtive)
    # print("y_relative: ", y_relative)
    # print("index_x: ", index_x)
    # print("index_y: ", index_y)

    return {"i_x": index_x, "i_y": index_y, "x_rel": x_realtive, "y_rel": y_relative, "w": w, "h": h}

def convert_to_trainingsdata():
    for img in images:
        sfw = img.get("sfw", None)
        sfh = img.get("sfh", None)
        array = np.zeros(shape=(row_size, row_size, 5))

        positions = img.get("positions", None)
        for position in positions:
            x = position.get("x", None)
            y = position.get("y", None)
            w = position.get("width", None)
            h = position.get("height", None)

            label_position = convert_coordinate_to_label(x, y ,w, h, sfw, sfh)
            array[label_position.get("i_x"), label_position.get("i_y"), 0] = 1
            array[label_position.get("i_x"), label_position.get("i_y"), 1] = label_position.get("x_rel") / sliding_window_size
            array[label_position.get("i_x"), label_position.get("i_y"), 2] = label_position.get("y_rel") / sliding_window_size
            array[label_position.get("i_x"), label_position.get("i_y"), 3] = label_position.get("w") / sliding_window_size
            array[label_position.get("i_x"), label_position.get("i_y"), 4] = label_position.get("h") / sliding_window_size
            
        samples.append(img.get("img"))
        labels.append(array)


convert_to_trainingsdata()

x_train = np.array(samples[:int(len(samples) / 2)])
del samples[:int(len(samples) / 2)]
x_test = np.array(samples)

del samples[:]

y_train = np.array(labels[:int(len(labels) / 2)])
del labels[:int(len(labels) / 2)]
y_test = np.array(labels)

del labels[:]


model = Sequential([
    # width x hight x dimension
    # 600x600x3
    Conv2D(20, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform", input_shape=(600, 600, 3)),
    # 596x596x20
    MaxPool2D((2,2)),
    # 298x298x20
    Conv2D(40, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 294x294x40
    Conv2D(40, 5, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 290x290x40
    MaxPool2D((3,3)),
    # 96x96x40
    Conv2D(80, 3, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 94x94x80
    MaxPool2D((2,2)),
    # 47x47x80
    Conv2D(160, 3, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 45x45x160
    Conv2D(5, 16, strides=(1, 1), data_format="channels_last", kernel_initializer="glorot_uniform"),
    # 30x30x5
    Dense(256, activation=None, use_bias = True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    # 30x30x5
    Dense(128, activation=None, use_bias = True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    # 30x30x5
    Dense(5, activation='sigmoid', use_bias = True, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
    # 30x30x5 --> Output
])

model.summary()

model.compile(
    #SGD(lr=0.001), 
    #SGD(lr=0.01),
    SGD(lr=0.01),
    loss="mean_squared_error",  
    metrics=[
        "accuracy"
    ]
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
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

predictions = model.predict(x_test)
# print(predictions)

for p in predictions:
    shape = p.shape
    
    for x in range(int(shape[0])):
        for y in range(int(shape[1])):
            if(p[x][y][0] > 0):
                print(p[x][y][0])
                print(p[x][y][1])
                print(p[x][y][2])
                print(p[x][y][3])
                print(p[x][y][4])


# exit()