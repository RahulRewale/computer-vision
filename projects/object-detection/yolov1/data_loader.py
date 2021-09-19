import tensorflow as tf
import tensorflow.keras as K
import pandas as pd
import numpy as np
import os



class DataLoader():

	# https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
	# 0 aeroplane; 1 bicycle; 2 bird; 3 boat; 4 bottle; 5 bus; 6 car; 7 cat; 8 chair; 9 cow; 
	# 10 diningtable; 11 dog; 12 horse; 13 motorbike; 14 person; 15 pottedplant; 16 sheep
	# 17 sofa; 18 train; 19 tvmonitor

	def __init__(self, image_dir, label_dir):
		""" Set the image and label directories
	
		Parameters:
		image_dir: the directory containing the input images
		label_dir: the directory containing the labels for the input images
		"""

		self.image_dir = image_dir
		self.label_dir = label_dir


	def load_train_data(self, csv_file, grid_size, classes, target_shape):
		"""
		Load image and corresponding label files

		Parameters:
		csv_file: files whose first column contains the image file names and the second column 
				  contains the corresponding label file names
		grdid_size: grid size to use
		classes: no. of classes
		target_shape: target shape for the images
		"""

		ds = tf.data.Dataset.from_generator(
					self.data_generator, args=(csv_file, grid_size, classes, target_shape),
					output_types = (tf.float32, tf.float32),			# for tf 2.3
					output_shapes = (target_shape, (grid_size, grid_size, classes+5))		# for tf 2.3
					# use below for tf 2.5+
					# output_signature=(
					# 		tf.TensorSpec(shape=target_shape),
					# 		tf.TensorSpec(shape=(grid_size, grid_size, classes+5))
					# 		)
					)

		return ds


	def data_generator(self, csv_file, grid_size, classes, target_shape):
		"""
		A generator that reads one image file and its corresponding label file and returns it

		Parameters:
		csv_file: files whose first column contains the image file names and the second column 
				  contains the corresponding label file names
		grdid_size: grid size to use
		classes: no. of classes
		target_shape: target shape for the images
		"""

		df = pd.read_csv(csv_file.decode('utf-8'), header=None)
		for index, row in df.iterrows():
			img = self.load_img(os.path.join(self.image_dir, row[0]))
			boxes = self.load_label(os.path.join(self.label_dir, row[1]), grid_size, classes, target_shape)
			yield tf.image.resize(img, target_shape[:2]), boxes


	def load_img(self, img_path):
		""" Load the image from img_path and normalize its pixels"""

		img = K.preprocessing.image.load_img(img_path)
		img_arr = K.preprocessing.image.img_to_array(img)
		img_arr /= 255.0
		return img_arr


	def load_label(self, label_path, grid_size, classes, target_shape):
		"""
		Load class labels and bounding boxes from the given file.
		Bounding boxes are given relative to the whole image, so the functino scales
		them as required
		
		Parameters:
		label_path: file name containing the classes and bounding boxes for a single image
		grid_size: grid size to use
		classes: object classes
		target_shape: target shape we want
		"""

		# load data from file
		boxes = np.loadtxt(label_path, ndmin=2)

		# create a numpy array to store the labels in the required format
		label = np.zeros((grid_size, grid_size, classes+5), dtype=np.float32)
		# The 5 in (classes + 5): prob(object), x-coord, y-coord, width, height

		# assuming same size for width and height of the image
		cell_size = target_shape[0]/grid_size

		for box in boxes:
			# obtain the actual mid-point coordinates
			box_mid_x_img = (box[1] * target_shape[1])
			box_mid_y_img = (box[2] * target_shape[0])

			# find the cell in which the midpoint resides
			cell_x = int(box_mid_x_img/cell_size)
			cell_y = int(box_mid_y_img/cell_size)
			
			# find the offset from the cell's top-left corner
			box_mid_x_cell = (box_mid_x_img - cell_size*cell_x)/cell_size
			box_mid_y_cell = (box_mid_y_img - cell_size*cell_y)/cell_size

			# normalize offset relative to the cell size
			box_width_cell = (box[3]*target_shape[1])/cell_size
			box_height_cell = (box[4]*target_shape[0])/cell_size

			# if there are two objects in a cell, we restrict it to only one
			if label[cell_x, cell_y, 20] == 0:
				label[cell_x, cell_y, int(box[0])] = 1
				label[cell_x, cell_y, 20:25] = [1, box_mid_x_cell, box_mid_y_cell, box_width_cell, box_height_cell]

		return label



if __name__ == "__main__":
	data_loader = DataLoader("voc_dataset\\images", "voc_dataset\\labels")
	train_ds = data_loader.load_train_data('.\\voc_dataset\\8examples.csv', 7, 20, (448, 448, 3))

	for image, label in train_ds:
		print(image.shape)
		print(label.shape)