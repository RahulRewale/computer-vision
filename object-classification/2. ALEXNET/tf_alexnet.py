''' This file contains TF 2 (using Keras API) implementation of AlexNet on Imagenette dataset
	Since, the local repose normalization layers in AlexNet are proven to be not that useful, 
	those layers are not included in this implementation.

	Dataset: Since ImageNet dataset is not publically available, I am using Imagenette dataset,
	which contains 10 classes.
'''

###########################################################################
########## THIS FILE IS OLD, SO IGNORE; USE ALEXNET.PY INSTEAD ##########
###########################################################################

import tensorflow as tf
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
# from tf_alexnet_utils import loadTrainingData


def loadData():
	''' Loads data for training, validation, and testing'''

	# use all the data in "train" folder for training
	# include patches of the original images
	# below function returns normalized data
	batchSize = 128
	# trainDS = loadTrainingData('datasets\\imagenette2-320\\train', batchSize, (227, 227))
	trainDS = tf.keras.preprocessing.image_dataset_from_directory(
			directory = 'datasets\\augmented',
			label_mode = 'categorical',
			batch_size = batchSize,
			image_size = (227, 227)		# TODO change to (256, 256) and then take 5 crops
			)

	
	# split the data in "val" folder into validation and testing dataset
	valTestDS = tf.keras.preprocessing.image_dataset_from_directory(
			directory = 'datasets\\imagenette2-320\\val',
			label_mode = 'categorical',
			batch_size = batchSize,
			image_size = (227, 227)		# TODO change to (256, 256) and then take 5 crops
			)

	noOfExamples = tf.data.experimental.cardinality(valTestDS).numpy()
	valDS = valTestDS.take(noOfExamples/2)
	testDS = valTestDS.skip(noOfExamples/2)

	# normalize images
	print(trainDS.element_spec)
	# print(tf.data.experimental.cardinality(trainDS.element_spec))
	trainDS = trainDS.map(lambda x, y: (x/255.0, y))
	valDS = valDS.map(lambda x, y: (x/255.0, y))
	testDS = testDS.map(lambda x, y: (x/255.0, y))

	return trainDS, valDS, testDS


def visualize(batchX, batchY, validLabels):
	''' Displays first 15 examples from the batch passed as argument'''
	for i in range(15):
		plt.subplot(3, 5, i+1)
		img = batchX[i]
		label = batchY[i]
		plt.imshow(img)
		plt.axis('off')
		index = np.argmax(label)
		plt.title(str(index) + " | " + str(validLabels[index]))
	plt.show()


def createAlexNet():
	''' Creates the AlexNet model without the Local Response Normalization layers'''
	inp = tf.keras.Input(shape=(227, 227, 3))
	
	flip = tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal') (inp)

	conv1 = tfl.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), 
				activation='relu', kernel_initializer='glorot_normal') (flip)
	# SKIPPED Local Response Normalization from Alex Net as it is not that useful
	pool1 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2)) (conv1)

	# pad1 = tfl.ZeroPadding2D(padding = (2, 2)) (pool1)
	conv2 = tfl.Conv2D(filters=256, kernel_size=(5, 5), padding='same', 
				activation='relu', kernel_initializer='glorot_normal') (pool1)
	# SKIPPED Local Response Normalization
	pool2 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))	(conv2)

	conv3 = tfl.Conv2D(filters=384, kernel_size=(3, 3), padding='same', 
				activation='relu', kernel_initializer='glorot_normal') (pool2)
	conv4 = tfl.Conv2D(filters=384, kernel_size=(3, 3), padding='same', 
				activation='relu', kernel_initializer='glorot_normal') (conv3)
	conv5 = tfl.Conv2D(filters=256, kernel_size=(3, 3), padding='same', 
				activation='relu', kernel_initializer='glorot_normal') (conv4)
	
	pool3 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))	(conv5)

	flat1 = tfl.Flatten() (pool3)
	fc1 = tfl.Dense(units=4096, activation='relu') (flat1)
	drop1 = tfl.Dropout(rate=0.5) (fc1)
	fc2 = tfl.Dense(units=4096, activation='relu') (drop1)
	drop2 = tfl.Dropout(rate=0.5) (fc2)
	
	# fc3 = tfl.Dense(units=1024, activation='relu') (drop2)
	# drop3 = tfl.Dropout(rate=0.3) (fc3)
	out = tfl.Dense(units=10, activation='softmax') (drop2)

	return tf.keras.Model(inputs=inp, outputs=out)


def predict(model, testDS, validLabels, savePath):
	print("*"*15 + "Testing"+ "*"*15 )
	print(model.evaluate(testDS))
	batchX, batchY = next(testDS.as_numpy_iterator())
	pred = model.predict(batchX)
	for i in range(15):
		plt.subplot(3, 5, i+1)
		plt.imshow(batchX[i])
		plt.title(validLabels[np.argmax(pred[i])])
		plt.axis('off')
	# plt.show()
	plt.tight_layout()
	plt.savefig(savePath + 'test-sample.png')




# Classes: https://github.com/fastai/imagenette#imagenette-1
validLabels = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
				'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']

# Load training dataset
trainDS, valDS, testDS = loadData()


# DEBUGGING: visualize few examples from the batch
# obtain an iterator over train_ds dataset
# niter = trainDS.as_numpy_iterator()
# pick the first batch from the training dataset
# trainBatchX, trainBatchY = next(niter)
# show some images
# visualize(trainBatchX, trainBatchY, validLabels)

# Create AlexNet model
model = createAlexNet()

# DEBUGGING: Check if all dimensions are as expected
# model.summary()

# Below doesn't seem to work well on my machine
# trainDS = trainDS.cache()#.prefetch(tf.data.experimental.AUTOTUNE)
# valDS = valDS.cache()#.prefetch(tf.data.experimental.AUTOTUNE)

# check if user has passed learning rate and optimizer to use
if len(sys.argv) == 3:
	learnRate = float(sys.argv[1])
	if sys.argv[2].lower() == 'sgd':
		optimizer = tf.keras.optimizers.SGD(learning_rate=learnRate, momentum=0.9)
		folderPath = f'SGD-LR-{learnRate}'
	elif sys.argv[2].lower() == 'adam':
		optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
		# Good learning rates for Adam: 0.0001, 0.00007, 0.00003
		folderPath = f'ADAM-LR-{learnRate}'
	else:
		optimizer = tf.keras.optimizers.RMSprop(learning_rate=learnRate)
		folderPath = f'RMS-LR-{learnRate}'
else:	# use default
	learnRate = 0.001
	optimizer = tf.keras.optimizers.SGD(learning_rate=learnRate, momentum=0.9)
	folderPath = f'SGD-LR-{learnRate}'


model.compile(optimizer= optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])


# create directories to store trainig history, tensorboard logs, model, and a test sample
checkPointPath = f'.\\checkpoints\\{folderPath}\\'
os.makedirs(checkPointPath)
os.makedirs(f'.\\tb-logs\\{folderPath}')

# callbacks
tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=f'.\\tb-logs\\{folderPath}')
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Reduce learning rate if validation loss doesn't improve much for two consecutive epochs
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=0.01, patience=2, min_lr=0.0001)

# train the model
print("*"*10, f"Using {model.optimizer.get_config()['name']} with learning Rate: {model.optimizer.get_config()['learning_rate']}", "*"*10)
model.fit(trainDS, epochs=20, validation_data=valDS, callbacks=[tensorBoard, reduceLR])

# save history to a csv file
df = pd.DataFrame(model.history.history)
# print(df)
df.to_csv(checkPointPath + 'history.csv')

# test the model on test data and save a sample
predict(model, testDS, validLabels, checkPointPath)

#save the model
model.save(checkPointPath + 'checkpoint')