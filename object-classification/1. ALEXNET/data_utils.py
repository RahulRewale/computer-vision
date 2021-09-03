import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# https://stackoverflow.com/questions/64326029/load-tensorflow-images-and-create-patches

def custom_extract_patches(batch_x, batch_y):
	""" 
		Creates patches for a batch of images
	
		Parameters:
		batch_x: batch of images
		batch_y: batch of labels for the above images
	"""

	patch_size = 227
	stride = 9			# chosen as per memory availability on my system
	patches = tf.image.extract_patches(batch_x, sizes=[1, patch_size, patch_size, 1], strides=[1, stride, stride, 1], rates=[1,1,1,1], padding='VALID')
	rowPatches = ((256 - patch_size)//stride) + 1
	no_patches = rowPatches*rowPatches	# no. of patches from a single image
	patches = tf.reshape(patches,  (-1,) + (patch_size, patch_size, 3))
	labels = tf.repeat(batch_y, repeats=no_patches, axis=0)
	labels = tf.reshape(labels, (-1,) + (10,))

	return patches, labels


def load_training_data(path, batch_size=32, image_size=(256, 256)):
	"""
		Loads images from given path, generates patches, and stores them

		Parameters:
		path: directory containing the images for which patches have to be created
		batch_size: batch size for loading images
		image_size: the size to which all images will be resized during loading
	"""

	# load images from given directory
	trainDS = tf.keras.preprocessing.image_dataset_from_directory(
				directory = path,
				label_mode = 'categorical',	
				batch_size = batch_size,	# using 32; fails for larger value because of memory constraints
				image_size = image_size,	# default
				# smart_resize = True, # available from v2.5
				)

	# For debugging
	# print(tf.data.experimental.cardinality(trainDS).numpy())
	# batchX, batchY = next(trainDS.as_numpy_iterator())
	# print(batchX.shape)
	# print(trainDS.element_spec)

	# generate patches
	trainDS = trainDS.map(custom_extract_patches)
	# unbatch, shuffle, and batch to make sure that patches from the same image are not together
	trainDS = trainDS.unbatch().shuffle(10000).batch(batchSize)

	# class-specific folders to store these patches
	# the folder names are same as that of the folders in "path" directory
	folderNames = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916',
					'n03417042', 'n03425413', 'n03445777', 'n03888257']

	#create folders
	for i in range(10):
		os.makedirs(f'..\\datasets\\augmented\\{folderNames[i]}')

	# save these patches to correct folders
	imgCount = 0
	batchNo = 0
	for batchX, batchY in trainDS:
		batchNo += 1
		print('Batch:', batchNo)
		for x, y in zip(batchX, batchY):
			imgCount += 1
			img = tf.keras.preprocessing.image.array_to_img(x)
			path = '..\\datasets\\augmented\\' + folderNames[np.argmax(y)] + "\\" + str(imgCount) + ".jpg"
			# print(y)
			# print(path)
			tf.keras.preprocessing.image.save_img(path, img)

	print("Total batches:", batchNo)
	print("Total images:", imgCount)


	# For visualizing few examples
	# batchX, batchY = next(trainDS.as_numpy_iterator())
	# validLabels = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
	# 			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']
	# for i in range(16):
	# 	plt.subplot(4, 4, i+1)
	# 	plt.imshow(batchX[i]/255.0)
	# 	plt.axis('off')
	# 	plt.title(validLabels[np.argmax(batchY[i])])
	# plt.show()

	return trainDS


if __name__ == "__main__":
	# tf.config.experimental.set_memory_growth(physical_devices[0], True)
	load_training_data('..\\datasets\\imagenette2-320\\train', 32, (227, 227))