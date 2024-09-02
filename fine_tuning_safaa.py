import matplotlib.pyplot as plt
import numpy as np
np.object = object
np.bool = bool
np.int = int
import os
import PIL
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

"""
hyper-params:
class name: this will be used as the class name and the folder name containing the data
img size : ex 256px x 256px
epochs
batch size
loss function
optimizer
metrics


preprocessing
check data imbalance
create folder with the spec names
split data into training and validation data
make sure all imgs are same size if not make the adequate changes
make sure the number of output in the final layer does match number of classes


"""

#
# data_dir = os.path.join('data','images')
# potato_early = list(glob.glob('data/Potato___Early_blight/*'))
# print(potato_early[0])
# PIL.Image.open(str(potato_early[0]))



img_height,img_width=180,180
batch_size=32
data_dir = 'data/'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names


resnet_model = Sequential()
#change num of classes here
pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
#change num of classes here
resnet_model.add(Dense(3, activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()


import cv2
potato_early = list(glob.glob('data/Potato___Early_blight/*'))
image=cv2.imread(str(potato_early[0]))
image_resized= cv2.resize(image, (img_height,img_width))
image=np.expand_dims(image_resized,axis=0)
pred=resnet_model.predict(image)
output_class=class_names[np.argmax(pred)]