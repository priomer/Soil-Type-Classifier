# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:47:26 2020

@author: ocn
"""


from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    
    return img_tensor


    
model = load_model("model_soil2.h5")
img_path = 'E:/AVRN_Report/Soil_Dataset/Test/Black_Soil/Black_41.jpg'
check_image = load_image(img_path)
prediction = model.predict(check_image)
print(prediction)

prediction =np.argmax(prediction, axis=1)
if prediction==0:
    prediction="Alluvial_Soil"
elif prediction==1:
    prediction="Black_Soil"
elif prediction==2:
    prediction="Clay_Soil"
elif prediction==3:
    prediction="Red_Soil"

print(prediction)    
    
    
