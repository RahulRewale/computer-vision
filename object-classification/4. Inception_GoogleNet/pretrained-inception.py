"""
This file loads pre-trained InceptionV3 model from tensorflow for classifying an image
"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt

inceptionModel = K.applications.InceptionV3(include_top=True, weights='imagenet')

img = K.preprocessing.image.load_img("..\\test-images\\all\\golf.jpeg", target_size=(299, 299))
arr = K.preprocessing.image.img_to_array(img)
inp = arr[np.newaxis, ...].copy()

newarr = K.applications.inception_v3.preprocess_input(inp)
pred = inceptionModel.predict(newarr)
label = K.applications.inception_v3.decode_predictions(pred, top=3)

plt.imshow(arr/255.0)
plt.axis('off')
plt.title(label[0][0])
plt.show()