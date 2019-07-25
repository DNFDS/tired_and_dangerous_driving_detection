import os
import cv2
import glob
import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

basedir = "ext/Data/distracted_driver_detection/"

model_image_size = 224

print("-------- loading train data")
X_train = list()
y_train = list()
for i in range(10):
    dir = os.path.join(basedir, "train", "c%d" % i)
    image_files = glob.glob(os.path.join(dir, "*.jpg"))
    print("loding {}, image count={}".format(dir, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X_train.append(cv2.resize(image, (model_image_size, model_image_size)))
        label = np.zeros(10, dtype=np.uint8)
        label[i] = 1
        y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
print("-------- loading valid data")
X_valid = list()
y_valid = list()
for i in range(10):
    dir = os.path.join(basedir, "valid", "c%d" % i)
    image_files = glob.glob(os.path.join(dir, "*.jpg"))
    print("loding {}, image count={}".format(dir, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X_valid.append(cv2.resize(image, (model_image_size, model_image_size)))
        label = np.zeros(10, dtype=np.uint8)
        label[i] = 1
        y_valid.append(label)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
base_model = VGG16(input_tensor=Input((model_image_size, model_image_size, 3)), weights='imagenet', include_top=False)

for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(10, activation='softmax')(x)
model = Model(base_model.input, x)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("done")
model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid))
model.save("models/vgg16-mymodel.h5")

model = load_model("models/vgg16-mymodel.h5")
print("load successed")

SVG(model_to_dot(model).create(prog='dot', format='svg'))
z = zip([x.name for x in model.layers], range(len(model.layers)))
for k, v in z:
    print("{} - {}".format(k,v))