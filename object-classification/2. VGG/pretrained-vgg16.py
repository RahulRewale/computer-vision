"""
This file loads pre-trained VGG16 model from tensorflow for classifying an image
"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt

vgg = K.applications.VGG16(include_top=True, weights='imagenet')

img = K.preprocessing.image.load_img('..\\test-images\\all\\golf.jpeg', target_size=(224, 224))
arr = K.preprocessing.image.img_to_array(img)
inp = arr[np.newaxis, ...].copy()

newarr = K.applications.vgg16.preprocess_input(inp)
pred = vgg.predict(newarr)
label = K.applications.vgg16.decode_predictions(pred, top=3)

# print(label)
plt.imshow(arr/255.0)
plt.axis('off')
plt.title(label[0][0])
plt.show()