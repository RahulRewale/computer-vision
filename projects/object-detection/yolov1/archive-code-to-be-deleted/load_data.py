import tensorflow as tf
import tensorflow.keras as K
import pandas as pd
import numpy as np



def generator(file_name, grid_size, classes, target_shape):
	df = pd.read_csv(file_name.decode('utf-8'), header=None)
	df[0] = ".\\voc_dataset\\images\\" + df[0]
	df[1] = ".\\voc_dataset\\labels\\" + df[1]
	for index, row in df.iterrows():
		img = load_img(row[0], target_shape[:2])
		boxes = load_label(row[1], grid_size, classes, img.shape, target_shape)
		yield tf.image.resize(img, target_shape[:2]), boxes


def load_label(label_path, S, classes, orig_shape, target_shape):

	boxes = np.loadtxt(label_path, ndmin=2)

	label = np.zeros((S, S, classes+5), dtype=np.float32)

	cell_height, cell_width = target_shape[0]/S, target_shape[1]/S

	width_factor = (target_shape[1]/orig_shape[1])
	height_factor = (target_shape[0]/orig_shape[0])

	for box in boxes:
		# find the cell the box belongs to
		box_mid_x_img = (box[1] * orig_shape[1]) * width_factor
		box_mid_y_img = (box[2] * orig_shape[0]) * height_factor

		cell_x = int(box_mid_x_img/cell_width)
		cell_y = int(box_mid_y_img/cell_height)
		
		box_mid_x_cell = (box_mid_x_img - cell_width*cell_x)/cell_width
		box_mid_y_cell = (box_mid_y_img - cell_height*cell_y)/cell_height

		box_width_cell = (box[3]*orig_shape[1])*width_factor/cell_width
		box_height_cell = (box[4]*orig_shape[0])*height_factor/cell_height

		# https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
		# 0 aeroplane; 1 bicycle; 2 bird; 3 boat; 4 bottle; 5 bus; 6 car; 7 cat; 8 chair; 9 cow; 
		# 10 diningtable; 11 dog; 12 horse; 13 motorbike; 14 person; 15 pottedplant; 16 sheep
		# 17 sofa; 18 train; 19 tvmonitor

		label[cell_x, cell_y, int(box[0])] = 1
		label[cell_x, cell_y, 20:25] = [1, box_mid_x_cell, box_mid_y_cell, box_width_cell, box_height_cell]

	return label


def load_img(img_path, target_shape):
	img = K.preprocessing.image.load_img(img_path)
	img_arr = K.preprocessing.image.img_to_array(img)
	img_arr /= 255.0
	return img_arr
	# return tf.image.resize(img_arr, target_shape)


def load_train_data(filePath, S, C, target_shape):
	ds = tf.data.Dataset.from_generator(generator, args=(filePath, S, C, target_shape), 
			output_types = (tf.float32, tf.float32),			# for tf 2.3
			output_shapes = (target_shape, (S, S, C+5)) )		# for tf 2.3

			# for tf 2.5+
			# output_signature=(tf.TensorSpec(shape=target_shape),tf.TensorSpec(shape=(, 7, 25))) )
	return ds



if __name__ == '__main__':
	ds = load_train_data('.\\voc_dataset\\8examples.csv', 7, 20, (448, 448, 3))

	for image, label in ds:
		print(image.shape)
		print(label.shape)