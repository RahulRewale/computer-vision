import tensorflow as tf
import tensorflow.keras as K
import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class CustomModel():
	""" This class creates a custom model from one of the pre-trained tensorflow models"""

	# a dictionary of supported models
	# key is the model name that we are using, and value is the name used for this model when it becomes part of a larger model
	validModels = {'vgg16':'vgg16', 'inceptionv3':'inception_v3', 'resnet50':'resnet50', 'mobilenet':'mobilenet_1.00_224'}


	def __init__(self, classes):
		"""Creates an object and sets the classes/categories for the dataset"""
		self.classes = classes


	def loadBaseModel(self, baseModelName='vgg16', shape=(224, 224, 3)):
		"""Loads a pre-trained model 
			Parameters:
			baseModelName: 	pass the name of the model to load; 
						the model name should be one of the supported models
			shape: pass the input shape for the model
		"""

		# check if the model is supported
		if baseModelName not in CustomModel.validModels:
			print(f"This model is not supported.\nSupported models are {CustomModel.validModels.keys()}\nExiting...")
			sys.exit(1)

		self.baseModelName = baseModelName
		
		# load the requested base model without its output layers
		if baseModelName == 'vgg16':
			self.baseModel = K.applications.VGG16(include_top=False, weights='imagenet', input_shape=shape)
		elif baseModelName == 'inceptionv3':
			self.baseModel = K.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=shape)
		elif baseModelName == 'resnet50':
			self.baseModel = K.applications.ResNet50(include_top=False, weights='imagenet', input_shape=shape)
		elif baseModelName == 'mobilenet':
			self.baseModel = K.applications.MobileNet(include_top=False, weights='imagenet', input_shape=shape)

		# freeze pre-trained base model weights
		self.baseModel.trainable = False
		# print(self.baseModel.summary())
		# save the base model graph just for verifiying
		K.utils.plot_model(self.baseModel, to_file=f'base-{baseModelName}.png', show_shapes=True)


	def addLayers(self, layersToAdd, inputShape=(224, 224, 3), preprocessInput=True, horizFlip=True):
		""" This method allows you to add any custom layers that you want to add
		on top of the loaded pre-trained model
		
		parameters:
			layersToAdd: a list of layers to add on top of the pre-trained base model
			inputShape: Keras.Input() will be created with this shape
			preprocessInput: Uses base model specific preprocessing method; pass False if not required
			horizFlip: If True, flips input image horizontaly with 50% probability
		"""

		# create input
		inp = K.Input(shape=inputShape)
		out = inp

		# preprocess using the base model specific function
		if preprocessInput:
			if self.baseModelName == 'vgg16':
				out = K.applications.vgg16.preprocess_input(inp)
			elif self.baseModelName == 'inceptionv3':
				out = K.applications.inception_v3.preprocess_input(inp)
			elif self.baseModelName == 'resnet50':
				out = K.applications.resnet.preprocess_input(inp)
			elif self.baseModelName == 'mobilenet':
				out = K.applications.mobilenet.preprocess_input(inp)

		# randomly horizontally flip images
		if horizFlip:
			out = K.layers.experimental.preprocessing.RandomFlip(mode='horizontal') (out)

		# pass images through the base model
		out = self.baseModel(out, training=False)	
		
		# https://github.com/keras-team/keras/pull/9965#issuecomment-382801648
		# setting training=False is required for BatchNorm layers
		# if you don't set this, during fine-tuning, mean and std will change according to the input
		# data despite setting trainable=False
		# thus, training=False is passed in the above call

		# pass the result through custom layers
		for layer in layersToAdd:
			out = layer(out)

		# create a model
		self.model = K.Model(inputs=inp, outputs=out)

		# plot a overall model graph for debugging
		K.utils.plot_model(self.model, to_file=f'custom-{self.baseModelName}.png', show_shapes=True)


	def printSummary(self):
		"""Displays base model and the custom model summaries"""
		print(self.baseModel.summary())
		print(self.model.summary())


	def loadData(self, batchSize=64, imageSize=(224, 224), trainLoc='datasets\\augmented', valLoc='datasets\\imagenette2-320\\val', testLoc=None):
		"""Loads training, validation, and testing data from given locations. If 
		testLoc is not given, splits the validation data into two parts
	
		Parameters:
		batchSize: Batch size to use for all the three datasets
		imageSize: Shape that we want for the loaded images
		trainLoc: Directory containing training data
		valLoc: Directory containing validation data
		testLoc: Directory containing testing data
		"""
		
		print(f'Loading training data from {trainLoc}')
		self.trainDS = K.preprocessing.image_dataset_from_directory(
				directory = trainLoc,
				label_mode = 'categorical', 
				batch_size = batchSize, 
				image_size = imageSize
				)

		print(f'Loading validation data from {valLoc}')
		self.valDS = K.preprocessing.image_dataset_from_directory(
					directory = valLoc,
					label_mode = 'categorical',
					batch_size = batchSize,
					image_size = imageSize
					)

		# if directory for test data is given, load test data from there
		# otherwise divide validation data into two equal parts
		if testLoc:
			print(f'Loading testing data from {testLoc}')
			self.testDS = K.preprocessing.image_dataset_from_directory(
					directory = testLoc,
					label_mode = 'categorical',
					batch_size = batchSize,
					image_size = imageSize
					)
		else:
			print('Splitting validation data')
			noOfExamples = tf.data.experimental.cardinality(self.valDS).numpy()
			self.testDS = self.valDS.skip(noOfExamples/2)		# last 50%
			self.valDS = self.valDS.take(noOfExamples/2)		# first 50%

		print("Training batches:", tf.data.experimental.cardinality(self.trainDS))
		print("Validation batches:", tf.data.experimental.cardinality(self.valDS))
		print("Testing batches:", tf.data.experimental.cardinality(self.testDS))
		return self.trainDS, self.valDS, self.testDS


	def trainModel(self, optimizer, loss, metrics, epochs=10, callBacks=None):
		"""Trains the newly added layers of the model using the given configuration.
		
		Parameters:
		optimizer:	pass the name of the optimizer to load with default configs 
				   	or pass a custom-built optimizer
		loss: 		pass the name of the loss function to use with default configs 
				   	or pass a custom-built loss function
		metrics: 	pass a list of metrics to use with default configs or pass 
					custom-built metrics
		epochs: 	number of epochs to train the model for
		callBacks:	any callbacks that you want to use
		"""
		
		# configure the model
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

		# train the model
		self.model.fit(self.trainDS, epochs=epochs, validation_data=self.valDS, callbacks=callBacks)


	def testModel(self, saveDir):
		"""Evaluates the model on the test dataset, displays a few test samples, and 
		finally saves the model
		
		parameters:
		saveDir:	location to save the trained model
		"""

		# create a directory for saving the model
		self.saveDir = saveDir
		# print(f'creating {saveDir}')
		os.makedirs(saveDir)

		trainResults = self.model.history.history
		# print(trainResults)
		print('Testing the model')
		testResults = self.model.evaluate(self.testDS)
		# print(testResults)


		# plot some samples from the one batch of the test data
		count = 9
		for x, y in self.testDS.take(1):	
			pass
		# x contains a batch of images
		# y contains a batch of labels

		pred = self.model.predict(x)	# predict on the batch

		for i, img in enumerate(x[:count]):	# take out the first 'count' images
			plt.subplot(3, 3, i+1)
			plt.imshow(img/255.0)
			plt.axis('off')
			plt.title(self.classes[np.argmax(pred[i])])

		
		# display figure
		fig = plt.gcf()
		plt.show()
		# save figure
		plt.draw()
		fig.savefig(self.saveDir + 'test.png')

		self.saveModel(trainResults, testResults)


	def saveModel(self, trainResults, testResults):
		"""	Saves the model and its training history in a csv file
			
			Parameters:
			trainResults: metrics generated during training
			testResults: metrics generated during tesing
		"""
		print("***********Saving model************")
		df_train = pd.DataFrame(trainResults)
		print(df_train)

		df_test = pd.DataFrame([testResults], columns=['test_loss', 'test_acc'])		
		print(df_test)
		
		df = pd.concat([df_train, df_test], axis=0)
		df.to_csv(self.saveDir + 'history.csv', index=False)
		self.model.save(self.saveDir + 'model')


	def loadModel(self, modelPath, baseModelName=None):
		"""
			Loads the model from the given path

			Parameters:
			modePath: Path of the model to load
			baseModelName: the base model used in this model; 
						   Used only when you want to fine tune the model;
						   Not used during testing/prediction
		"""

		completePath = f'.\\models\\{modelPath}\\model'
		print(f'Loading model from location {completePath}')
		self.model = K.models.load_model(completePath)
		
		# used only during fine-tuning
		# during testing, not required
		if baseModelName: 
			self.baseModel = self.model.get_layer(CustomModel.validModels[baseModelName])


	def setTrainableLayers(self, layerNo):
		"""
			Sets few layers to be trainable in the base model; used during fine-tuning
			Parameters:
			layerNo: the layer location (counted from the end) in the base model from which to train
		"""

		self.baseModel.trainable = True
		
		print(f"Training from layer {self.baseModel.layers[-layerNo].name} onwards")

		# only last "layerNo" layers are to be trained; rest to be set to trainable=False
		for layer in self.baseModel.layers[:-layerNo]:
			layer.trainable = False


	def predict(self, testDir, inputShape=(224, 224)):
		"""
			Tests the model on the images in the dirctory passed as parameter
			Parameters:	
			testDir:	the directory containing the test images
			inputShape: shape of images to be tested
		"""

		testDS = K.preprocessing.image_dataset_from_directory(
								directory = testDir,
								label_mode=None,
								image_size = inputShape,
								shuffle=False
							)

		for x in testDS.take(1):
			pass

		pred = self.model.predict(x)
		for i, img in enumerate(x):
		 	plt.subplot(3, 3, i+1)
		 	plt.imshow(img/255.0)
		 	plt.axis('off')
		 	plt.title(self.classes[np.argmax(pred[i])])

		plt.show()



# Moved this code to model specific files
# if __name__ == '__main__':

# 	# classes/categories for our dataset
# 	classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
# 			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']

# 	# create a CustomModel object
# 	customModel = CustomModel(classes)

# 	# check what we want to do: train, tune, or test
# 	if sys.argv[1].lower() == 'train':
# 		baseModelName = sys.argv[2]
		
# 		# Load the pre-trained base model
# 		customModel.loadBaseModel(baseModelName, shape=(227, 227, 3))
		
# 		# add custom layers to the model

# 		# Using Flatten Layer
# 		# customModel.addLayers(
# 		# 		[
# 		# 		K.layers.Flatten(), 
# 		# 		K.layers.Dense(units=1024, activation='relu'), 
# 		# 		K.layers.Dropout(rate=0.6),
# 		# 		K.layers.Dense(units=1024, activation='relu'),
# 		# 		K.layers.Dropout(rate=0.6),
# 		# 		K.layers.Dense(units=10, activation='softmax')
# 		# 		],
# 		# 		inputShape = (227, 227, 3)
# 		# 	)

# 		# Using Global Average Pooling Layer
# 		customModel.addLayers([
# 				K.layers.GlobalAveragePooling2D(),
# 				K.layers.Dense(units=512, activation='relu'),
# 				K.layers.Dropout(0.3),
# 				K.layers.Dense(units=512, activation='relu'),
# 				K.layers.Dropout(0.3),
# 				K.layers.Dense(units=10, activation='softmax')
# 				],
# 				inputShape = (227, 227, 3)
# 				)

# 		# customModel.printSummary()

# 		# load datasets
# 		customModel.loadData(imageSize=(227, 227))

# 		# train model with required config
# 		learnRate  = float(sys.argv[3])
# 		optimizer=K.optimizers.SGD(learning_rate=learnRate)
# 		earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# 		customModel.trainModel(optimizer, 'categorical_crossentropy', 
# 								['accuracy'], 10, [earlyStop])

# 		# test model
# 		customModel.testModel(f'.\\models\\{baseModelName}\\trainedmodel-{learnRate}\\')
		
# 	elif sys.argv[1].lower() == 'tune':
# 		baseModelName = sys.argv[2]
# 		baseLearnRate = float(sys.argv[3])
# 		learnRate = float(sys.argv[4])
# 		layerNo = int(sys.argv[5])
		
# 		# load the trained model using the model name and learning rate
# 		customModel.loadModel(f'{baseModelName}\\trainedmodel-{baseLearnRate}', baseModelName)

		
# 		# set the trainable layers in the model
# 		customModel.setTrainableLayers(layerNo)

# 		# load datasets
# 		customModel.loadData(imageSize=(227, 227))

# 		# tune the model with the required config
# 		optimizer=K.optimizers.SGD(learning_rate=learnRate)
# 		earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# 		customModel.trainModel(optimizer, 'categorical_crossentropy', 
# 								['accuracy'], 10, [earlyStop])

# 		# test the model
# 		customModel.testModel(f'.\\models\\{baseModelName}\\tunedModel-{baseLearnRate}-{learnRate}-{layerNo}\\')

# 	elif sys.argv[1].lower() == 'test':
# 		modelName = sys.argv[2]
# 		baseLearnRate = float(sys.argv[3])
# 		tuningLearnRate = float(sys.argv[4])
# 		layerNo = int(sys.argv[5])
# 		testDir = sys.argv[6]

# 		# Load the fine-tuned model
# 		customModel.loadModel(f'{modelName}\\tunedModel-{baseLearnRate}-{tuningLearnRate}-{layerNo}')

# 		# test the model
# 		customModel.predict(testDir, inputShape=(227, 227))