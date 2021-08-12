import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# https://stackoverflow.com/questions/64326029/load-tensorflow-images-and-create-patches

def extractPatches(batchX, batchY):
	batchX = batchX/255.0
	patchSize = 227
	stride = 9			# chosen as per memory availability on my system
	patches = tf.image.extract_patches(batchX, sizes=[1, patchSize, patchSize, 1], strides=[1, stride, stride, 1], rates=[1,1,1,1], padding='VALID')
	rowPatches = ((256 - patchSize)//stride) + 1
	noOfPatches = rowPatches*rowPatches	# no. of patches from a single image
	patches = tf.reshape(patches,  (-1,) + (patchSize, patchSize, 3))
	labels = tf.repeat(batchY, repeats=noOfPatches, axis=0)
	labels = tf.reshape(labels, (-1,) + (10,))

	return patches, labels


def loadTrainingData(path, batchSize=32, imageSize=(256, 256)):

	trainDS = tf.keras.preprocessing.image_dataset_from_directory(
				directory = path,
				label_mode = 'categorical',	
				batch_size = 32,			# default; fails for larger value because of memory constraints
				image_size = (256, 256),	# default
				# smart_resize = True, # available from v2.5
				)

	# For debugging
	# print(tf.data.experimental.cardinality(trainDS).numpy())
	# batchX, batchY = next(trainDS.as_numpy_iterator())
	# print(batchX.shape)
	# print(trainDS.element_spec)

	# generate patches
	trainDS = trainDS.map(extractPatches)
	# unbatch, shuffle, and batch to make sure that patches from the same image are not together
	trainDS = trainDS.unbatch().shuffle(10000).batch(batchSize)


	folderNames = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916',
					'n03417042', 'n03425413', 'n03445777', 'n03888257']

	trainDS.class_names
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
	# 	plt.imshow(batchX[i])
	# 	plt.axis('off')
	# 	plt.title(validLabels[np.argmax(batchY[i])])
	# plt.show()

	# Returning small amount of data; remove take() in future
	return trainDS


if __name__ == "__main__":
	# physical_devices = tf.config.list_physical_devices('GPU')
	# print("************************")
	# print(physical_devices[0], "memory growth set to True")
	# tf.config.experimental.set_memory_growth(physical_devices[0], True)
	# print("************************")
	loadTrainingData('..\\datasets\\imagenette2-320\\train', 64, (227, 227))