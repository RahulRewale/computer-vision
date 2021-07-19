import tensorflow as tf
import tensorflow.keras as K
from custom_model import CustomModel 
import sys


# classes/categories for our dataset
classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
		'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']

# create a CustomModel object
customModel = CustomModel(classes)
baseModelName = 'inceptionv3'

# check what we want to do: train, tune, or test
if sys.argv[1].lower() == 'train':
	
	# Load the pre-trained base model
	customModel.loadBaseModel(baseModelName, shape=(224, 224, 3))
	
	# add custom layers to the model

	# Using Flatten Layer
	# customModel.addLayers(
	# 		[
	# 		K.layers.Flatten(), 
	# 		K.layers.Dense(units=512, activation='relu'),
	# 		K.layers.Dropout(rate=0.6),
	# 		K.layers.Dense(units=10, activation='softmax')
	# 		],
	# 		inputShape = (224, 224, 3)
	# 	)

	# Using Global Average Pooling Layer
	customModel.addLayers([
			K.layers.GlobalAveragePooling2D(),
			K.layers.Dense(units=256, activation='relu'),
			K.layers.Dropout(0.3),
			K.layers.Dense(units=10, activation='softmax')
			],
			inputShape = (224, 224, 3)
			)

	# customModel.printSummary()

	# load datasets
	customModel.loadData(imageSize=(224, 224), trainLoc='datasets\\manually-created')	# default imageSize = (224, 224)

	# train model with required config
	learnRate  = float(sys.argv[2])
	optimizer=K.optimizers.SGD(learning_rate=learnRate, momentum=0.9)
	earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
	customModel.trainModel(optimizer, 'categorical_crossentropy', 
							['accuracy'], 5, [earlyStop])

	# test model
	customModel.testModel(f'.\\models\\{baseModelName}\\trainedmodel-{learnRate}\\')
	
elif sys.argv[1].lower() == 'tune':
	baseLearnRate = float(sys.argv[2])
	learnRate = float(sys.argv[3])
	layerNo = int(sys.argv[4])
	
	# load the trained model using the model name and learning rate
	customModel.loadModel(f'{baseModelName}\\trainedmodel-{baseLearnRate}', baseModelName)

	
	# set the trainable layers in the model
	customModel.setTrainableLayers(layerNo)

	# load datasets
	customModel.loadData(imageSize=(224, 224), trainLoc='datasets\\manually-created')

	# tune the model with the required config
	optimizer=K.optimizers.SGD(learning_rate=learnRate)
	earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
	customModel.trainModel(optimizer, 'categorical_crossentropy', 
							['accuracy'], 5)

	# test the model
	customModel.testModel(f'.\\models\\{baseModelName}\\tunedModel-{baseLearnRate}-{learnRate}-{layerNo}\\')

elif sys.argv[1].lower() == 'test':
	baseLearnRate = float(sys.argv[2])
	tuningLearnRate = float(sys.argv[3])
	layerNo = int(sys.argv[4])
	testDir = sys.argv[5]

	# Load the fine-tuned model
	customModel.loadModel(f'{baseModelName}\\tunedModel-{baseLearnRate}-{tuningLearnRate}-{layerNo}')

	# test the model
	customModel.predict(testDir, inputShape=(224, 224))
