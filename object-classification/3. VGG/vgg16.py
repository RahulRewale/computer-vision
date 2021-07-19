import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl
import sys

def createModel(input_shape):
	x = K.Input(shape=input_shape)

	# preprocessing and augmentation
	# out = tfl.experimental.preprocessing.CenterCrop(height=224, width=224) (x)
	# out = tfl.experimental.preprocessing.Rescaling(1/255.0) (x)
	# out = tfl.experimental.preprocessing.RandomFlip(mode='horizontal') (out)
	

	# model layers
	out = tfl.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
			activation='relu') (x)
	out = tfl.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.MaxPooling2D(pool_size=2, strides=2) (out)

	out = tfl.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.MaxPooling2D(pool_size=2, strides=2) (out)

	out = tfl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)		
	out = tfl.MaxPooling2D(pool_size=2, strides=2) (out)

	out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)
	out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
			activation='relu') (out)		
	out = tfl.MaxPooling2D(pool_size=2, strides=2) (out)

	# out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
	# 		activation='relu') (out)
	# out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
	# 		activation='relu') (out)
	# out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
	# 		activation='relu') (out)		
	# out = tfl.MaxPooling2D(pool_size=2, strides=2) (out)


	out = tfl.Flatten() (out)
	out = tfl.Dense(units=512, activation='relu')(out)
	out = tfl.Dropout(rate=0.3) (out)
	out = tfl.Dense(units=512, activation='relu')(out)
	out = tfl.Dropout(rate=0.3) (out)
	y = tfl.Dense(units=10, activation='softmax')(out)

	return K.Model(inputs=x, outputs=y)


def configure(model, learnRate):
	model.compile(optimizer=K.optimizers.SGD(learning_rate=learnRate, momentum=0.9),
					loss= K.losses.CategoricalCrossentropy(),
					metrics = ['accuracy']
				)


def train(model, trainDS, valDS, epochs=10):
	# csvlogger = K.callbacks.CSVLogger("training.csv")
	# learnScheduler = K.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2)

	model.fit(x=trainDS, epochs=epochs, validation_data=valDS)


def test(model, testDS):
	print(model.evaluate(testDS))


def saveModel(model, file_name):
	model.save('.\\trained_model\\' + file_name)


def loadData(input_shape):
	trainDS = K.preprocessing.image_dataset_from_directory(
				directory="..\\datasets\\augmented", label_mode='categorical',
				image_size=input_shape, batch_size=32, seed=11, validation_split=0.2, 
				subset='training'
			)
	trainDS = trainDS.map(lambda x, y: (x/255.0, y))
	

	valDS = K.preprocessing.image_dataset_from_directory(
				directory="..\\datasets\\augmented", label_mode='categorical',
				image_size=input_shape, batch_size=32, seed=11, validation_split=0.2, 
				subset='validation'
			)
	valDS = valDS.map(lambda x, y: (x/255.0, y))

	return trainDS.take(30), valDS.take(10)


def loadTestData(input_shape):
	testDS = K.preprocessing.image_dataset_from_directory(
				directory="..\\datasets\\imagenette2-320\\val", label_mode='categorical',
				image_size=input_shape, batch_size=16
			)

	return testDS.take(30)

	

if __name__ == "__main__":
	input_shape = (224, 224, 3)
	learning_rate = float(sys.argv[1])	# 0.005
	epochs = int(sys.argv[2])			# 30
	model = createModel(input_shape)
	K.utils.plot_model(model, 'my_vgg16.png', show_shapes=True, show_layer_names=True)

	configure(model, learning_rate)
	trainDS, valDS= loadData(input_shape[:2])
	train(model, trainDS, valDS, epochs)
	saveModel(model, f"{learning_rate}-{epochs}")
	test(model, loadTestData(input_shape[:2]))