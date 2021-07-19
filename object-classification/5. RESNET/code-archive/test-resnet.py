import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import glob


def loadData():
	trainDS = K.preprocessing.image_dataset_from_directory(
			directory = '..\\datasets\\augmented',
			label_mode = 'categorical', 
			batch_size = 64, 
			image_size = (227, 227)
			)

	valTestDS = K.preprocessing.image_dataset_from_directory(
				directory = '..\\datasets\\imagenette2-320\\val',
				label_mode = 'categorical',
				batch_size = 64,
				image_size = (227, 227)
				)

	noOfExamples = tf.data.experimental.cardinality(valTestDS).numpy()
	valDS = valTestDS.take(noOfExamples/2)		# first 50%
	testDS = valTestDS.skip(noOfExamples/2)		# remaining 50%

	return trainDS, valDS, testDS


modelPath = sys.argv[1]
classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']

model = tf.keras.models.load_model(modelPath+"\\model")
# trainDS, valDS, testDS = loadData()


jpegFiles = glob.glob('test-images\\*.jpeg')
for i in range(len(jpegFiles)):
	print(jpegFiles[i])
	img = K.preprocessing.image.load_img(jpegFiles[i], target_size=(227, 227))
	arr = K.preprocessing.image.img_to_array(img)
	arr = arr[np.newaxis, ...]
	
	pred = model.predict(arr)
	print("*"*10, pred, "*"*10)

	plt.subplot(3, 3, i+1)
	plt.imshow(arr[0]/255.0)
	plt.axis('off')
	plt.title(classes[np.argmax(pred[0])])

plt.show()