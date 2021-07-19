import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt

mobileNet = K.applications.MobileNet(include_top=True, weights='imagenet')

img = K.preprocessing.image.load_img('..\\test-images\\all\\dog.jpeg', target_size=(224, 224))
arr = K.preprocessing.image.img_to_array(img)
inp = arr[np.newaxis, ...].copy()

newarr = K.applications.mobilenet.preprocess_input(inp)
pred = mobileNet.predict(newarr)
label = K.applications.mobilenet.decode_predictions(pred, top=3)

plt.imshow(arr/255.0)
plt.axis('off')
plt.title(label[0][0])
plt.show()