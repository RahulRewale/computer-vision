import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

img = tf.keras.preprocessing.image.load_img("ILSVRC2012_val_00046669.jpeg")
img_arr = tf.keras.preprocessing.image.img_to_array(img)
img_new_arr = tf.image.resize(img_arr, (256, 256))
imgs_arr = img_new_arr[np.newaxis, ...]

raw_patches = tf.image.extract_patches(imgs_arr, [1, 227, 227, 1], [1, 29, 29, 1], [1,1,1,1], "VALID")
patches = tf.reshape(raw_patches, (-1,)+(227, 227, 3))

# fig, axes = plt.subplots(2, 2)
# axes[0, 0].imshow(patches[0]/255.0)
# axes[0, 1].imshow(patches[1]/255.0)
# axes[1, 0].imshow(patches[2]/255.0)
# axes[1, 1].imshow(patches[3]/255.0)
plt.imshow(patches[0]/255.0)
plt.show()

plt.imshow(patches[1]/255.0)
plt.show()

plt.imshow(patches[2]/255.0)
plt.show()

plt.imshow(patches[3]/255.0)
plt.show()

central_patches = tf.image.resize(tf.image.central_crop(imgs_arr, 0.88), (227, 227))
print(central_patches.shape)
plt.imshow(central_patches[0]/255.0)
plt.show()