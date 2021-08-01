import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class DataLoader():
	"""Class to load training, validation, and testing data"""

	def __init__(self, train_dir=None, val_dir=None, test_dir=None, validation_split=0.1, batch_size=128):
		""" Sets the locations from where data should be loaded
	
		Parameters:
		train_dir: directory containing the training data
		val_dir: directory containing the validation data
		test_dir: directory containing the testing data
		validation_split: if val_dir is not provided, the split size will be used
		batch_size: size of each batch
		"""

		self.train_dir = train_dir
		
		if not val_dir:
			self.val_dir = self.train_dir
		else:
			self.val_dir = val_dir

		self.test_dir = test_dir
		self.validation_split = validation_split
		self.batch_size = batch_size


	def load_train_val_data(self, input_shape=(227, 227), train_dir=None, val_dir=None):
		"""Loads training and validation data

		Parameter:
		input_shape: target image size that we want for images
		train_dir: if None, self.train_dir will be used; otherwise, train_dir will be used
		val_dir: if None, self.val_dir will be used; otherwise, val_dir will be used
		"""

		# if path is given, use it
		if train_dir:
			self.train_dir = train_dir
		if val_dir: 
			self.val_dir = val_dir

		# if both paths are same, we need to split data into two parts
		if self.train_dir == self.val_dir:
			# load training data
			train_ds = K.preprocessing.image_dataset_from_directory(
					directory=self.train_dir, label_mode='categorical',
					image_size=input_shape, batch_size=self.batch_size, seed=11, 
					validation_split=self.validation_split, subset='training'
				)

			# load validation data
			val_ds = K.preprocessing.image_dataset_from_directory(
					directory=self.val_dir, label_mode='categorical',
					image_size=input_shape, batch_size=self.batch_size, seed=11, 
					validation_split=self.validation_split, subset='validation'
				)
		else:
			# load training data
			train_ds = K.preprocessing.image_dataset_from_directory(
					directory=self.train_dir, label_mode='categorical',
					image_size=input_shape, batch_size=self.batch_size
				)

			# load validation data
			val_ds = K.preprocessing.image_dataset_from_directory(
					directory=self.val_dir, label_mode='categorical',
					image_size=input_shape, batch_size=self.batch_size
				)

		# normalize pixel values
		self.train_ds = train_ds.map(lambda x, y: (x/255.0, y))
		self.val_ds = val_ds.map(lambda x, y: (x/255.0, y))

		return self.train_ds, self.val_ds


	def load_test_data(self, input_shape=(227, 227), test_dir=None):
		"""Loads test data

		Parameter:
		input_shape: target image size that we want for images
		test_dir: if None, self.test_dir will be used; otherwise, test_dir will be used
		"""

		if test_dir:
			self.test_dir = test_dir

		# load test data
		test_ds = K.preprocessing.image_dataset_from_directory(
				directory=self.test_dir, label_mode='categorical',
				image_size=input_shape, batch_size=self.batch_size
			)

		# normalize pixel values
		self.test_ds = test_ds.map(lambda x, y: (x/255.0, y))
		
		return self.test_ds



if __name__ == "__main__":

	ds_loader = DataLoader("..\\datasets\\imagenette2-320\\train",
							"..\\datasets\\imagenette2-320\\train",
							"..\\datasets\\imagenette2-320\\val")
	train_ds, val_ds = ds_loader.load_train_val_data()
	test_ds = ds_loader.load_test_data()

	print(tf.data.experimental.cardinality(train_ds))
	print(tf.data.experimental.cardinality(val_ds))
	print(tf.data.experimental.cardinality(test_ds))
