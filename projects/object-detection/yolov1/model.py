import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl
import numpy as np
import os


# YOLO model config;
# [no. of filters, filter size, stride]
# M - means MaxPool with pool size 2*2 and stride 2

model_config = [
	(64, 7, 2),
	"M",
	(192, 3, 1),
	"M",
	(128, 1, 1),
	(256, 3, 1),
	(256, 1, 1),
	(512, 3, 1),
	"M",
	(256, 1, 1),
	(512, 3, 1),
	(256, 1, 1),
	(512, 3, 1),
	(256, 1, 1),
	(512, 3, 1),
	(256, 1, 1),
	(512, 3, 1),
	(512, 1, 1),
	(1024, 3, 1),
	"M",
	(512, 1, 1),
	(1024, 3, 1),
	(512, 1, 1),
	(1024, 3, 1),
	(1024, 3, 1),
	(1024, 3, 2),
	(1024, 3, 1),
	(1024, 3, 1)
]



class YOLOV1(K.Model):
	"""
	Class to create the YOLO network
	"""

	def __init__(self, yolo_input_shape, grid_size, boxes, classes):
		""" Set few model parameters

		Parameters:
		yolo_input_shape: input shape to the model
		grid_size: grid size to use
		boxes: no. of boxes per grid cell
		classes: no. of classes
		"""

		super().__init__()
		self.yolo_input_shape = yolo_input_shape
		self.grid_size = grid_size
		self.boxes = boxes
		self.classes = classes
		self.model_config = model_config
		self.create_model()


	def create_model(self):
		"""
		Create the model using the model_config
		"""

		inp = K.Input(self.yolo_input_shape)
		out = inp

		for entry in self.model_config:
			if type(entry) == str:
				out = tfl.MaxPool2D(pool_size=(2, 2), strides=2) (out)
			elif entry[1] == 1:
				out = tfl.Conv2D(filters=entry[0], kernel_size=(entry[1], entry[1]), strides=entry[2],
						activation=tfl.LeakyReLU(alpha=0.1)) (out)
			else:
				out = tfl.Conv2D(filters=entry[0], kernel_size=(entry[1], entry[1]), strides=entry[2], 
						activation=tfl.LeakyReLU(alpha=0.1), padding='same') (out)

		out = tfl.Flatten() (out)
		
		# YOLO model uses 4096 neurons with 0.5 dropout
		# Using smaller value due to memory constraints
		out = tfl.Dense(units=1024, activation=tfl.LeakyReLU(alpha=0.1)) (out)
		out = tfl.Dropout(0.2) (out)

		out = tfl.Dense(units=self.grid_size*self.grid_size*(self.classes+self.boxes*5)) (out)
		y = tfl.Reshape((self.grid_size, self.grid_size, self.classes+self.boxes*5)) (out)

		self.yolo_model = K.Model(inp, y)
		

	def call(self, input, training=False):
		""" Execute the model """
		return self.yolo_model(input, training)


	def save_yolo(self, path, optimizer=None):
		self.save(os.path.join(path, "yolo-model"))
		if optimizer:
			np.save(os.path.join(path, "opt-weights.npy"), optimizer.get_weights())


	def load_yolo(self, path, optimizer=None):


		# https://stackoverflow.com/a/64671177
		self.yolo_model = K.models.load_model(os.path.join(path, "yolo-model"))
		
		if optimizer:
			opt_weights = np.load(os.path.join(path, "opt-weights.npy"), allow_pickle=True)
			
			# dummy zero gradients
			zero_grads = [tf.zeros_like(w) for w in self.yolo_model.trainable_variables]

			# save current state of variables
			saved_vars = [tf.identity(w) for w in self.yolo_model.trainable_variables]

			# Apply gradients which don't do anything with Adam
			optimizer.apply_gradients(zip(zero_grads, self.yolo_model.trainable_variables))

			# Reload variables
			[x.assign(y) for x, y in zip(self.yolo_model.trainable_variables, saved_vars)]

			# Set the weights of the optimizer
			optimizer.set_weights(opt_weights)
		


if __name__ == '__main__':
	model = YOLOV1((448, 448, 3), 7, 2, 20)
	out = model(np.random.random((2, 448, 448, 3)))
	assert out.shape == (None, 7, 7, 30)