# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:36:12 2020

@author: ocn
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# for plotting images (optional)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    

# getting data
base_dir = 'E:\AVRN_Report\Soil_Dataset'
train_dir = os.path.join(base_dir, 'Train')
val_dir = os.path.join(base_dir, 'Test')

train_Alluvial_Soil = os.path.join(train_dir, 'Alluvial_Soil')
train_Black_soil = os.path.join(train_dir, 'Black_Soil')
train_Clay_Soil = os.path.join(train_dir, 'Clay_Soil')
train_Red_Soil = os.path.join(train_dir, 'Red_Soil')


valid_Alluvial_Soil = os.path.join(val_dir, 'Alluvial_Soil')
valid_Black_soil = os.path.join(val_dir, 'Black_Soil')
valid_Clay_Soil = os.path.join(val_dir, 'Clay_Soil')
valid_Red_Soil = os.path.join(val_dir, 'Red_Soil')


num_Alluvial_Soil_tr = len(os.listdir(train_Alluvial_Soil))
num_Black_Soil_tr = len(os.listdir(train_Black_soil))
num_Clay_Soil_tr = len(os.listdir(train_Clay_Soil))
num_Red_Soil_tr = len(os.listdir(train_Red_Soil))

num_Alluvial_Soil_val = len(os.listdir(valid_Alluvial_Soil))
num_Black_Soil_val = len(os.listdir(valid_Black_soil))
num_Clay_Soil_val = len(os.listdir(valid_Clay_Soil))
num_Red_Soil_val = len(os.listdir(valid_Red_Soil))




total_train = num_Alluvial_Soil_tr + num_Black_Soil_tr + num_Clay_Soil_tr + num_Red_Soil_tr

total_val = num_Alluvial_Soil_val + num_Black_Soil_val + num_Clay_Soil_val + num_Red_Soil_val 

BATCH_SIZE = 32
IMG_SHAPE = 200 # square image


#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=val_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')
images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(images)


model = Sequential()
# Conv2D : Two dimentional convulational model.
# 32 : Input for next layer
# (3,3) convulonational windows size
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SHAPE, IMG_SHAPE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5)) # half of neurons will be turned off randomly to prevent overfitting
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Activation('relu')) 
        
# output dense layer; since thenumbers of classes are 10 here so we need to pass 
#minimum 10 neurons whereas 2 in cats and dogs   
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


EPOCHS = 10

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )


# analysis of the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# save model and architecture to single file
model.save("model_soil1.h5")
print("Saved soil classification model to disk")
