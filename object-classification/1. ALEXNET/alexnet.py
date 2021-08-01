import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl


class AlexNet(K.models.Model):

	def __init__(self, dense_units=512, drop=0.6, weight_decay=0.008, classes=10):

		super().__init__()

		self.flip = tfl.experimental.preprocessing.RandomFlip(mode='horizontal')

		# random rotation
		# rot = tfl.experimental.preprocessing.RandomRotation(factor=0.2) (flip)

		self.conv1 = tfl.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), 
					activation='relu', kernel_initializer='glorot_normal', 
					kernel_regularizer=K.regularizers.L2(weight_decay))
		# Skipped Local Response Normalization layer from AlexNet as it is proven to be not that useful
		self.pool1 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

		self.conv2 = tfl.Conv2D(filters=256, kernel_size=(5, 5), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay))
		# Skipped Local Response Normalization layer
		self.pool2 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

		self.conv3 = tfl.Conv2D(filters=384, kernel_size=(3, 3), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay))
		self.conv4 = tfl.Conv2D(filters=384, kernel_size=(3, 3), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay))
		self.conv5 = tfl.Conv2D(filters=256, kernel_size=(3, 3), padding='same', 
					activation='relu', kernel_initializer='glorot_normal',
					kernel_regularizer=K.regularizers.L2(weight_decay))
		self.pool3 = tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2))


		# FC layers with dropout
		self.flat = tfl.Flatten()
		self.dense1 = tfl.Dense(units=dense_units, activation='relu')
		self.drop1 = tfl.Dropout(rate=drop)
		self.dense2 = tfl.Dense(units=dense_units, activation='relu')
		self.drop2 = tfl.Dropout(rate=drop)
		
		# output layer
		self.classifier = tfl.Dense(units=classes, activation='softmax')


	def call(self, inputs, training=False):

		out = self.flip(inputs)
		out = self.conv1(out)
		out = self.pool1(out)
		out = self.conv2(out)
		out = self.pool2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.pool3(out)
		out = self.flat(out)
		out = self.dense1(out)
		out = self.drop1(out)
		out = self.dense2(out)
		out = self.drop2(out)
		return self.classifier(out)
