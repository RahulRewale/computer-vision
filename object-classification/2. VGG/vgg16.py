import tensorflow.keras as K
import tensorflow.keras.layers as tfl
import numpy as np


class VGG16():

	def __init__(self):
		pass

	def create(self, input_shape=(224, 224, 3), dense_units=1024, drop=0.5, classes=10):

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

		out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu') (out)
		out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu') (out)
		out = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu') (out)		
		out = tfl.MaxPooling2D(pool_size=2, strides=2) (out)

		out = tfl.Flatten() (out)
		out = tfl.Dense(units=dense_units, activation='relu')(out)
		out = tfl.Dropout(rate=drop) (out)
		out = tfl.Dense(units=dense_units, activation='relu')(out)
		out = tfl.Dropout(rate=drop) (out)
		y = tfl.Dense(units=classes, activation='softmax')(out)

		self.model = K.Model(inputs=x, outputs=y)
		return self.model


	def configure(self, optimizer=K.optimizers.SGD(learning_rate=0.005, momentum=0.9), 
					loss= K.losses.CategoricalCrossentropy(),
					metrics=['accuracy']):
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


	def train(self, train_ds, val_ds, epochs=10, callbacks=None):
		self.model.fit(x=train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)


	def test(self, test_ds):
		print(self.model.evaluate(test_ds))


	def predict(self, x):
		return self.model.predict(x)


	def plot_model(self, path):
		K.utils.plot_model(self.model, path, show_shapes=True)


	def save_model(self, path):
		model.save(path)




if __name__ == "__main__":
	inp = np.random.random((5, 224, 224, 3))
	vgg_model = VGG16()
	vgg_model.create()
	out = vgg_model.predict(inp)
	print(out.shape)
	vgg_model.plot_model('mymodel.png')