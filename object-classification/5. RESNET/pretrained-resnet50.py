import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt

resenet = K.applications.ResNet50(include_top=True, weights='imagenet')

img = K.preprocessing.image.load_img('..\\test-images\\all\\golf.jpeg', target_size=(224, 224))
arr = K.preprocessing.image.img_to_array(img)
inp = arr[np.newaxis, ...].copy()

newarr = K.applications.resnet.preprocess_input(inp)
pred = resenet.predict(newarr)
label = K.applications.resnet.decode_predictions(pred, top=3)

plt.imshow(arr/255.0)
plt.axis('off')
plt.title(label[0][0])
plt.show()