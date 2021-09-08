import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
	"""Class to load training, validation, and testing data"""

	def __init__(self, train_dir=None, val_dir=None, test_dir=None, validation_split=0.3, batch_size=64):
		""" Sets the directories from where data should be loaded
	
		Parameters:
		train_dir: directory containing the training data
		val_dir: directory containing the validation data
		test_dir: directory containing the testing data
		validation_split: if val_dir is not provided, the split size will be used
		batch_size: size of each batch
		"""

		self.train_dir = train_dir
		
		if not val_dir:
			self.val_dir = test_dir
		else:
			self.val_dir = val_dir

		self.test_dir = test_dir
		self.validation_split = validation_split
		self.batch_size = batch_size


	def load_val_test_data(self, input_shape=(256, 256), val_dir=None, test_dir=None):
		"""Loads testing and validation data

		Parameter:
		input_shape: target image size that we want for images
		test_dir: if None, self.test_dir will be used; otherwise, test_dir will be used
		val_dir: if None, self.val_dir will be used; otherwise, val_dir will be used
		"""

		# if path is given, use it
		if test_dir:
			self.test_dir = test_dir
		if val_dir: 
			self.val_dir = val_dir
		# if both paths are same, we need to split data into two parts

		final_val_x = []
		final_val_y = []
		final_test_x = []
		final_test_y = []

		# load data
		all_ds = K.preprocessing.image_dataset_from_directory(
					directory=self.test_dir, label_mode='categorical',
					image_size=input_shape, batch_size=self.batch_size
				)

		total_size = tf.data.experimental.cardinality(all_ds) 
		print("all_ds size:", total_size)

		if self.test_dir == self.val_dir:
			# split data into validation and testing set
			test_ds = all_ds.take(total_size.numpy() * self.validation_split)
			val_ds = all_ds.skip(total_size.numpy() * self.validation_split)
		else:
			test_ds = all_ds
			# load val_ds from given directory
			val_ds = K.preprocessing.image_dataset_from_directory(
					directory=self.val_dir, label_mode='categorical',
					image_size=input_shape, batch_size=self.batch_size
				)

		print("val_ds size:", tf.data.experimental.cardinality(val_ds))
		print("test_ds size:", tf.data.experimental.cardinality(test_ds))

		# normalize pixel values
		self.test_ds = test_ds.map(lambda x, y: (x/255.0, y))
		self.val_ds = val_ds.map(lambda x, y: (x/255.0, y))

		# for batch_x, batch_y in self.val_ds.take(1):
		# 	print(batch_x.shape, batch_y.shape)
		# 	for x, y in zip(batch_x, batch_y):
		# 		print(x.shape)
		# 		print(y.shape)
		# 		plt.imshow(x)
		# 		plt.title(np.argmax(y))
		# 		plt.show()

		# TODO - combine above to map calls with the below ones
		# just divide x by 255.0 before passing to self.generate_crops()

		# generate 5 crops per image
		self.val_ds = self.val_ds.map(lambda x, y: (self.generate_testing_crops(x, y)))
		self.test_ds = self.test_ds.map(lambda x, y: (self.generate_testing_crops(x, y)))
		
		# seems the images get shuffled during map() because the first image in the orignal
		# dataset and the first image in the patched dataset are different
		# nonetheless, the labels for patches are correct. So, ignoring it.
		
		# for batch_x, batch_y in self.val_ds.take(1):
		# 	print(batch_x.shape, batch_y.shape)
		# 	for x, y in zip(batch_x, batch_y):
		# 		print(x.shape, y.shape)
		# 		fig, axes = plt.subplots(2, 5)
		# 		for i in range(2):
		# 			for j in range(5):
		# 				axes[i,j].imshow(x[i*5+j])
		# 				axes[i,j].set_title(np.argmax(y))
		# 				axes[i,j].axis("off")
		# 		plt.show()

		return self.val_ds, self.test_ds


	def generate_testing_crops(self, batch_x, batch_y):
		return self.generate_crops(batch_x), batch_y


	def generate_crops(self, batch_x):
		# take 4 patches from corners of the image of size (256, 256)
		raw_patches = tf.image.extract_patches(batch_x, sizes=[1, 227, 227, 1], 
						strides=[1, 29, 29, 1], rates=[1,1,1,1], padding="VALID")
		patches = tf.reshape(raw_patches, (-1,)+(227, 227, 3))
		flipped_patches = tf.image.flip_left_right(patches)
		
		patches = tf.reshape(patches, (-1, 4) + (227, 227, 3))
		flipped_patches = tf.reshape(flipped_patches, (-1, 4) + (227, 227, 3))


		# take central crop from the image
		central_patches = tf.image.resize(tf.image.central_crop(batch_x, 0.88), (227, 227))
		flipped_central_patches = tf.image.flip_left_right(central_patches)

		#increase dimensions by one to be able to concatenate
		central_patches = central_patches[:, tf.newaxis, ...]
		flipped_central_patches = flipped_central_patches[:, tf.newaxis, ...]

		# concatenate above to create 10 patches per image
		final_patches = tf.concat((patches, central_patches, flipped_patches, flipped_central_patches), axis=1)

		return final_patches


	def generate_training_crops(self, batch_x, batch_y):
		labels = tf.repeat(batch_y, repeats=10, axis=0)
		return tf.reshape(self.generate_crops(batch_x), (-1, 227, 227, 3)), labels


	def load_train_data(self, input_shape=(256, 256), train_dir=None):
		"""Loads training data

		Parameter:
		input_shape: target image size that we want for images
		train_dir: if None, self.train_dir will be used; otherwise, train_dir will be used
		"""

		if train_dir:
			self.train_dir = train_dir

		# load data
		train_ds = K.preprocessing.image_dataset_from_directory(
				directory=self.train_dir, label_mode='categorical',
				image_size=input_shape, batch_size=self.batch_size
			)

		print("train_ds size:", train_ds.cardinality())

		# normalize pixel values
		self.train_ds = train_ds.map(lambda x, y: (x/255.0, y))
		
		#flipped_ds = self.train_ds.map(lambda x, y: (tf.image.flip_left_right(x), y))
		#self.train_ds = self.train_ds.concatenate(flipped_ds)
		self.train_ds = self.train_ds.map(lambda x,y: self.generate_training_crops(x,  y))
		#self. train_ds = tf.reshape(self.train_ds, (-1, 227, 227, 3))
		
		self.train_ds = self.train_ds.unbatch().shuffle(5000).batch(self.batch_size)

		return self.train_ds



if __name__ == "__main__":

	ds_loader = DataLoader("..\\..\\datasets\\imagenette2-320\\train",
							"..\\..\\datasets\\imagenette2-320\\val",
							"..\\..\\datasets\\imagenette2-320\\val")
	train_ds = ds_loader.load_train_data()
	val_ds, test_ds = ds_loader.load_val_test_data()

	print(tf.data.experimental.cardinality(train_ds))
	print(tf.data.experimental.cardinality(val_ds))
	print(tf.data.experimental.cardinality(test_ds))


	for batch_x, batch_y in train_ds.take(1):
		print(batch_x.shape)
		print(batch_y.shape)

		for x, y in zip(batch_x, batch_y):
			plt.imshow(x)
			plt.title(np.argmax(y))
			plt.axis('off')
			plt.show()