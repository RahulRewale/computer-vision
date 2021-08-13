import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl


class VGG16(K.models.Model):
	"""This class extends Keras Model class and creates a VGG16 model"""


	def __init__(self, dense_units=512, drop=0.6, weight_decay=0.008, classes=10):
		"""
			Creates all the required layers using Keras
			Parameters:
			dense_units: no. of units in the dense layers
			drop: dropout for the dense layers
			weight_decay: weight decay for all conv. layers
			classes: no. of categories/classes of objects
		"""

		super().__init__()

		# preprocessing and augmentation
		# self.centercrop = tfl.experimental.preprocessing.CenterCrop(height=224, width=224)
		# self.scale = tfl.experimental.preprocessing.Rescaling(1/255.0)
		self.flip = tfl.experimental.preprocessing.RandomFlip(mode='horizontal')
		self.rotate = tfl.experimental.preprocessing.RandomRotation(factor=0.2)
		# self.contrast = tfl.RandomContrast(factor=0.2)


		# conv block 1 layers
		self.block1_conv1 = tfl.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block1_conv2 = tfl.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block1_pool = tfl.MaxPooling2D(pool_size=2, strides=2)

		# conv block 2 layers
		self.block2_conv1 = tfl.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block2_conv2 = tfl.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block2_pool = tfl.MaxPooling2D(pool_size=2, strides=2)

		# conv block 3 layers
		self.block3_conv1 = tfl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block3_conv2 = tfl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block3_conv3 = tfl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block3_pool = tfl.MaxPooling2D(pool_size=2, strides=2)

		# conv block 4 layers
		self.block4_conv1 = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block4_conv2 = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block4_conv3 = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block4_pool = tfl.MaxPooling2D(pool_size=2, strides=2)

		# conv block 5 layers
		self.block5_conv1 = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block5_conv2 = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block5_conv3 = tfl.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', 
				activation='relu')
		self.block5_pool = tfl.MaxPooling2D(pool_size=2, strides=2)

		# flatten and dense layers
		self.flat = tfl.Flatten()
		self.dense1 = tfl.Dense(units=dense_units, activation='relu')
		self.drop1 = tfl.Dropout(rate=drop)
		self.dense2 = tfl.Dense(units=dense_units, activation='relu')
		self.drop2 = tfl.Dropout(rate=drop)
		
		# output layer layers
		self.classifier = tfl.Dense(units=classes, activation='softmax')


	def call(self, inputs, training=False, augmentation=True):
		"""Passes inputs through the alexnet layers"""

		out = inputs

		if augmentation:
			out = self.flip(out)
			out = self.rotate(out)

		# conv block 1 layers
		out = self.block1_conv1(inputs)
		out = self.block1_conv2(out)
		out = self.block1_pool(out)

		# conv block 2 layers
		out = self.block2_conv1(out)
		out = self.block2_conv2(out)
		out = self.block2_pool(out)

		# conv block 3 layers
		out = self.block3_conv1(out)
		out = self.block3_conv2(out)
		out = self.block3_conv3(out)
		out = self.block3_pool(out)

		# conv block 4 layers
		out = self.block4_conv1(out)
		out = self.block4_conv2(out)
		out = self.block4_conv2(out)
		out = self.block4_pool(out)

		# conv block 5 layers
		out = self.block5_conv1(out)
		out = self.block5_conv2(out)
		out = self.block5_conv3(out)
		out = self.block5_pool(out)

		# flatten and dense layers
		out = self.flat(out)
		out = self.dense1(out)
		out = self.drop1(out)
		out = self.dense2(out)
		out = self.drop2(out)

		# output
		return self.classifier(out)

