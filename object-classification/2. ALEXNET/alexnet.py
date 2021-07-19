''' This file contains TF 2 (using Keras API) implementation of AlexNet on "Imagenette" dataset
	with minor modifications.

	AlexNet Paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
	
	Dataset: Since "ImageNet" dataset is not publically available, this code uses "Imagenette" dataset,
	which can be found at https://github.com/fastai/imagenette
'''

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# By mistake, dropout for all of these was disabled
# Using 4096 neurons
# (weight_decay, dropout, lr, train_acc, val_acc)
# (0.02, 0.6, 0.001, 0.9779, 0.6776)
# (0.025, 0.6, 0.001, 0.7872, 0.6505)
# (0.03, 0.6, 0.001, 0.8064, 0.6620)
# (0.04, 0.6, 0.001, 0.7106, 0.6594)
# (0.05, 0.6, 0.001, 0.7074, 0.6146)
# (0.045, 0.6, 0.001, 0.6857, 0.6255)

# Using 512 neurons
# (0.005, 0.6, 0.001, 0.995, 0.72)
# (0.01, 0.6, 0.001, 0.9969, 0.7042)
# (0.05, 0.6, 0.001, 0.6256, 0.5984)
# (0.02, 0.6, 0.001, 0.92, 0.67)
# (0.03, 0.6, 0.001, 0.8237, 0.6839)


# Using 256 neurons
# (0.01, 0.5, 0.001, 0.9802, 0.7198)
# (0.02, 0.5, 0.001, 0.8936, 0.7182)
# (0.05, 0.5, 0.001, ???) # cancelled



# with dropout enabled
# (weight_decay, dropout, lr, train_acc, val_acc)

# Using 512 neurons (600 batches)
# (0.0001, 0.2, 0.05, 0.1037, 0.0885) -- no learning at all

# Using 1024 neurons (600 batches)
# (0.0001, 0.2, 0.05, inf, inf)
# (0.0001, 0.2, 0.03, 0.9688, 0.7505)
# (0.0001, 0.4, 0.03, 0.1012, 0.0828) -- no learning at all
# (0.0001, 0.3, 0.03, 0.9892, 0.7812)	******
# (0.0001, 0.35, 0.03, 0.9762, 0.7635)
# (0.0003, 0.3, 0.03, 0.9836, 0.7719)
# (0.0008, 0.3, 0.03, 0.9761, 0.7693)
# (0.001, 0.3, 0.03, 0.9761, 0.7464)
# (0.0008, 0.35, 0.03, ) -- no learning
# (0.003, 0.3, 0.03, 0.9591, 0.7625)
# (0.009, 0.3, 0.03, 0.8398, 0.7369)
# (0.005, 0.3, 0.03, 0.8741, 0.7119)
# (0.007, 0.3, 0.03, 0.8635, 0.6844)

# Using 2048 neurons (600 batches)
# (0.0001, 0.3, 0.03, 0.9977, 0.7755)
# (0.0003, 0.3, 0.03, 0.9983, 0.7745)
# (0.0008, 0.3, 0.03, 0.9875, 0.7740) 
# (0.001, 0.3, 0.03, 0.9971, 0.7797)
# (0.005, 0.3, 0.03, 0.9659, 0.7229)
# (0.005, 0.4, 0.03, 0.9659, 0.7229)
# (0.004, 0.6, 0.03, ) -- no learning at all
# (0.00001, 0.6, 0.03, 0.9931, 0.7953)
# (0.0001, 0.6, 0.03, 0.9960, 0.8141)	******
# (0.001, 0.6, 0.05, 0.9790, 0.8125)
# (0.001, 0.7, 0.05, ...) -- no learning at all
# (0.005, 0.6, 0.05, ) -- no learning
# (0.003, 0.6, 0.05, ) -- no learning
# (0.0008, 0.5, 0.03, 0.9872, 0.7974)
# (0.0008, 0.6, 0.03, 0.9790, 0.8078)
# (0.001, 0.6, 0.03, 0.9857, 0.8135)
# (0.001, 0.6, 0.03, 0.9613, 0.8125)


# Using 2048 neurons (all batches)
# (0.001, 0.6, 0.03, ) -- no learning
# (0.001, 0.6, 0.05, ) -- no learning
# (0.001, 0.6, 0.09, ) -- no learning
# (0.001, 0.6, 0.01, 0.9947, 0.8281) // factor 0.2
# (0.005, 0.6, 0.01, 0.9398, 0.8109) //factor 0.2
# (0.008, 0.6, 0.01, 0.9423, 0.8229) //factor 0.2
# (0.01, 0.6, 0.01, 0.9186, 0.8115) //factor 0.3
# (0.03, 0.6, 0.01, 0.5769, 0.5734) //factor 0.3 -- slow learning


# Using 4096 neurons
# (0.01, 0.6, 0.01, ) //factor 0.3



class AlexNet():
	''' This class creates AlexNet model for object classification
		It also contains methods for loading data
	'''

	def __init__(self, input_shape):

		# input layer
		inp = K.Input(shape=input_shape)

		weight_decay = 0.01
		drop = 0.6

		# AlexNet paper mentions that the authors used image patches and their horizontal reflections
		# for training. Here, we are using either the input patch or its horizontal reflection randomly
		flip = tfl.experimental.preprocessing.RandomFlip(mode='horizontal') (inp)

		conv1 = tfl.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), 
					activation='relu', kernel_initializer='glorot_normal', 
					kernel_regularizer=K.regularizers.L2(weight_decay)) (flip)
		# Skipped Local Response Normalization layer from AlexNet as it is proven to be not that useful
		pool1 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2)) (conv1)

		conv2 = tfl.Conv2D(filters=256, kernel_size=(5, 5), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay)) (pool1)
		# Skipped Local Response Normalization layer
		pool2 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))	(conv2)

		conv3 = tfl.Conv2D(filters=384, kernel_size=(3, 3), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay)) (pool2)
		conv4 = tfl.Conv2D(filters=384, kernel_size=(3, 3), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay)) (conv3)
		conv5 = tfl.Conv2D(filters=256, kernel_size=(3, 3), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay)) (conv4)
		pool3 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))	(conv5)	

		# FC layers with dropout
		out = tfl.Flatten() (pool3)
		out = tfl.Dense(units=4096, activation='relu') (out)
		out = tfl.Dropout(rate=drop) (out)
		out = tfl.Dense(units=4096, activation='relu') (out)
		out = tfl.Dropout(rate=drop) (out)
		
		# output layer
		y = tfl.Dense(units=10, activation='softmax') (out)

		self.model = K.Model(inputs=inp, outputs=y)

		K.utils.plot_model(self.model, to_file="model.png", show_shapes=True)


	def loadTrainData(self, path='..\\datasets\\augmented', input_shape=(227, 227), batchSize=128):
		'''This file loads images from the given path for training'''
		trainDS = K.preprocessing.image_dataset_from_directory(
			directory = path, 
			label_mode = 'categorical', 
			batch_size = batchSize, 
			image_size = input_shape
			)

		# normalize pixels
		self.trainDS = trainDS.map(lambda x, y: (x/255.0, y))
		# self.trainDS = self.trainDS.take(600)
		
		# self.trainDS = self.trainDS.cache() # OOM on my system

		print("Training data loaded successfully")
		print(tf.data.experimental.cardinality(self.trainDS))
		return self.trainDS


	def loadValTestData(self, path='..\\datasets\\imagenette2-320\\val', input_shape=(227, 227), batchSize=128):
		'''This file loads images from the given path and splits those into validation and testing data'''

		valTestDS = K.preprocessing.image_dataset_from_directory(
			directory = path,
			label_mode = 'categorical',
			batch_size = batchSize,
			image_size = input_shape
			)

		noOfExamples = tf.data.experimental.cardinality(valTestDS).numpy()
		valDS = valTestDS.take(noOfExamples/2)		# first 50%
		testDS = valTestDS.skip(noOfExamples/2)	# remaining 50%


		# normalize pixels
		self.valDS = valDS.map(lambda x, y: (x/255.0, y))
		self.testDS = testDS.map(lambda x, y: (x/255.0, y))

		# self.valDS = self.valDS.cache()
		# self.testDS = self.testDS.cache()
		print("Validation and testing data loaded successfully")

		# self.valDS = self.valDS.take(10)
		# self.testDS = self.testDS.take(10)

		return self.valDS, self.testDS


	def setLabels(self, class_names):
		'''Class names for the images'''
		
		self.classes= class_names


	def compile(self, optimizer=K.optimizers.SGD(learning_rate=0.003), 
				loss=K.losses.CategoricalCrossentropy(), metrics=['accuracy']):
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


	def alexnet_summary(self):
		print("\n\nModel Summary")
		print(self.model.summary())
		print(self.model.optimizer.get_config())


	def train(self, logPath, epochs=2, trainDataset=None, valDataset=None):

		if trainDataset == None:
			trainDataset = self.trainDS
		
		if valDataset == None:
			valDataset = self.valDS

		# create folders for logging
		checkPointPath = f'.\\checkpoints\\{logPath}\\'
		tbLogsPath = f'.\\tb-logs\\{logPath}'
		os.makedirs(checkPointPath)
		os.makedirs(tbLogsPath)

		# callbacks
		tensorBoard = K.callbacks.TensorBoard(log_dir=tbLogsPath)
		# earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
		# Reduce learning rate if validation loss doesn't improve much for two consecutive epochs
		reduceLR = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, min_delta=0.05, 
												patience=2, cooldown=1, min_lr=0.0001, verbose=1)
		ckpt = K.callbacks.ModelCheckpoint(
					checkPointPath + 'checkpoint-epoch{epoch:02d}-loss{val_loss:.3f}',
					save_weights_only=True, monitor='val_accuracy', 
					mode='max', save_best_only=True)
		callbacks = [tensorBoard, reduceLR, ckpt]

		print("\n\n", "*"*15, f"Training model with {self.model.optimizer.get_config()['name']} and learning Rate {self.model.optimizer.get_config()['learning_rate']}", "*"*15)
		history = self.model.fit(trainDataset, epochs=epochs, validation_data=valDataset, callbacks=callbacks)

		# save model training history to a csv file
		df = pd.DataFrame(history.history)
		df.to_csv(checkPointPath + 'history.csv')

		# save the model
		# self.model.save(checkPointPath + 'checkpoint')
		print("Model trained successfully")
		return history


	def testModel(self, logPath, dataset=None):
		'''This method evaluates the model on the given dataset and stores a test sample 
		to given path'''

		checkPointPath = f'.\\checkpoints\\{logPath}\\'

		print('\n\n', "*"*15, "Testing model", "*"*15)
		if dataset == None:
			dataset = self.testDS
		
		metrics = self.model.evaluate(dataset, return_dict=True)
		print(metrics)

		# take out a single batch from the dataset
		batchX, batchY = next(dataset.as_numpy_iterator())
		pred = self.model.predict(batchX)
		for i in range(15):	# plot few images from the batch
			plt.subplot(3, 5, i+1)
			plt.imshow(batchX[i])
			plt.title(self.classes[np.argmax(pred[i])])
			plt.axis('off')
		
		# plt.show()
		plt.tight_layout()
		# save the figure
		plt.savefig(checkPointPath + 'test-sample.png')

		return metrics


	def predict(self, dataset):
		return self.model.predict(dataset)


	def loadWeights(self, path):
		self.model = K.models.load_model(path)


	def loadModel(self, path):
		self.model.load_weights(path)


if __name__ == "__main__":
	# Classes: https://github.com/fastai/imagenette#imagenette-1
	classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
				'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']
	
	alexnet = AlexNet((227, 227, 3))
	# alexnet.alexnet_summary()
	alexnet.loadTrainData()
	alexnet.loadValTestData()
	alexnet.setLabels(classes)
	# alexnet.compile()

	# check if user has passed optimizer, learning rate, and epochs to use
	# pass all values or none
	if len(sys.argv) >= 4:
		learnRate = float(sys.argv[2])
		epochs = int(sys.argv[3])
		if sys.argv[1].lower() == 'sgd':
			optimizer = tf.keras.optimizers.SGD(learning_rate=learnRate, momentum=0.9)
			logFolder = f'SGD-LR-{learnRate}'
		elif sys.argv[1].lower() == 'adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
			# Good learning rates for Adam: 0.0001, 0.00007, 0.00003
			logFolder = f'ADAM-LR-{learnRate}'
		else:
			optimizer = tf.keras.optimizers.RMSprop(learning_rate=learnRate)
			logFolder = f'RMS-LR-{learnRate}'
	else:	# use default
		learnRate = 0.01
		optimizer = tf.keras.optimizers.SGD(learning_rate=learnRate, momentum=0.9)
		epochs = 15
		logFolder = f'SGD-LR-{learnRate}'


	if len(sys.argv) == 5:
		print("Loading pre-trained model")
		alexnet.loadModel(sys.argv[4])
	
	alexnet.compile(optimizer)

	# model summary
	# alexnet.alexnet_summary()

	history = alexnet.train(logFolder, epochs)

	metrics = alexnet.testModel(logFolder)