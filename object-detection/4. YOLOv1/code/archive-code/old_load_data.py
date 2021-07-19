import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from matplotlib import patches 
import pandas as pd
import os
import numpy as np



def load_train_ds():
	df = pd.read_csv('.\\voc_dataset\\train.csv', header=None)
	df = df.iloc[:10]
	df[0] = ".\\voc_dataset\\images\\" + df[0]
	df[1] = ".\\voc_dataset\\labels\\" + df[1]
	print(df)
	data = df.to_numpy()
	train_files_ds = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))
	# for x, y in train_files_ds:
	# 	print(x, y)

	# seems tf.py_function adds one extra dimension
	# So, need to take the first element using [0]
	# ds = train_files_ds.map(lambda x,y: (tf.py_function(load_img, [x], [tf.float32])[0], tf.py_function(load_label, [y], [tf.float32])[0]))
	ds = train_files_ds.map(lambda x,y: (load_img(x), tf.py_function(load_label, [y], [tf.float32])[0]))
	# To test that the data is loaded correctly, display image and plot bounding boxes on top
	# for x, y in ds:
	# 	disp_image(x, y.numpy()[:, 1:])

	return ds


def load_img(img_path):
	# img = K.preprocessing.image.load_img(os.path.join("voc_dataset", "images", img_path.numpy().decode('utf-8')))
	img = K.preprocessing.image.load_img(img_path)
	img_arr = K.preprocessing.image.img_to_array(img)
	img_arr /= 255.0
	return img_arr


def load_label(label_path):
	# path = os.path.join("voc_dataset", "labels", label_path.numpy().decode('utf-8'))
	boxes = np.loadtxt(label_path, ndmin=2)
	return boxes


def disp_image(img, boxes):
	plt.imshow(img)
	ax = plt.gca()
	for y in boxes:
		# multiply bounding box x-coordinate and width by image width
		y[[0, 2]] *= img.shape[1]
		# multiply bounding box y-coordinate and height by image height
		y[[1, 3]] *= img.shape[0]
		
		# since Rectangle() takes coordinates of a corner, and we have coordinates of
		# mid-point, subtract half of bounding-box width and height from x and y coordinates
		rect = patches.Rectangle((y[0]-y[2]/2, y[1]-y[3]/2), width=y[2], height=y[3], linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
	
	plt.show()


def transform_data(ds, target_shape=(448, 448), S=7, classes=20):

	ds = ds.map(lambda x, y: (tf.py_function(tf.image.resize, [x, target_shape], [tf.float32])[0], 
		# transform_label(y, target_shape, S, classes)))
		tf.py_function(transform_label, [y, target_shape, S, classes], tf.float32)))
	

	for x, y in ds:
		print(x.shape, y.shape)
		disp_image(x.numpy(), y.numpy()[:, 1:])
	
	return ds


def transform_label(boxes, img_shape, S, classes):

	label = np.zeros((S, S, classes+5), dtype=np.float32)

	#img_shape = img_shape.numpy().astype(np.float32)
	cell_size = img_shape[0]//S, img_shape[1]//S
	# cell_size = cell_size.numpy().astype(np.float32)
	print(cell_size)

	for box in boxes:
		# find the cell the box belongs to
		box_img_mid_x = box[1] * img_shape[1]
		box_img_mid_y = box[2] * img_shape[0]

		cell_x = box_img_mid_x//cell_size
		cell_y = box_img_mid_y//cell_size

		box_mid_cell_x = (box_img_mid_x - cell_size*cell_x)/cell_size
		box_mid_cell_y = (box_img_mid_y - cell_size*cell_y)/cell_size

		box_width_cell = (box[3]*img_shape[1])/cell_size
		box_height_cell = (box[4]*img_shape[0])/cell_size

		# https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
		# 0 aeroplane; 1 bicycle; 2 bird; 3 boat; 4 bottle; 5 bus; 6 car; 7 cat; 8 chair; 9 cow; 
		# 10 diningtable; 11 dog; 12 horse; 13 motorbike; 14 person; 15 pottedplant; 16 sheep
		# 17 sofa; 18 train; 19 tvmonitor

		label[int(cell_x), int(cell_y), :21] = box[:21]
		label[cell_x, cell_y, 21:] = [box_mid_cell_x, box_mid_cell_y, box_width_cell, box_height_cell]

	return label

	# orig_shape = np.array(x.shape.as_list()).astype(np.float32)
	# target_shape = np.array(target_shape).astype(np.float32)
	# print(orig_shape)
	# print(target_shape)
	# # new_y = np.array(y.shape)
	# print(y.numpy())
	# new_y = y.numpy()
	# # https://stackoverflow.com/questions/59652250/after-cropping-a-image-how-to-find-new-bounding-box-coordinates
	# new_y[:, 1]	= y[:, 1] - ( (orig_shape[1] - target_shape[1])/2)
	# # new_y[:, 3]	= y[:, 3] - ( (orig_shape[1] - target_shape[1])/2)
	# new_y[:, 2]	= y[:, 2] - ( (orig_shape[0] - target_shape[0])/2)
	# # new_y[:, 4]	= y[:, 4] - ( (orig_shape[0] - target_shape[0])/2)

	# print(new_y)

	# # new_y[:, 3]	= (y[:, 3]/orig_shape[1]) * target_shape[1]
	# # new_y[:, 2]	= (y[:, 2]/orig_shape[0]) * target_shape[0]
	# # new_y[:, 4]	= (y[:, 4]/orig_shape[0]) * target_shape[0]
	return new_y


if __name__ == "__main__":
	load_train_ds()




# def load_data():
# 	img_ds = K.preprocessing.image_dataset_from_directory(
# 				directory='.\\voc_dataset\\images', 
# 				labels=None,
# 				batch_size=16,
# 				image_size=(448, 448),
# 				shuffle=False
# 			)

# 	label_ds = K.preprocessing.text_dataset_from_directory(
# 				directory='.\\voc_dataset\\labels', 
# 				labels=None,
# 				batch_size=16,
# 				shuffle=False
# 			)

# 	return img_ds, label_ds



# img_ds, label_ds = load_data()

# for img_batch, label_batch in zip(img_ds.take(1), label_ds.take(1)):
# 	for x, y in zip(img_batch, label_batch):	# interate over each (image, label)
# 		x = x/255.0
# 		out = tf.strings.split(y)
# 		boxes = out.numpy().astype(float)
# 		boxes = boxes.reshape((-1, 5))
# 		boxes[:, 1:] = boxes[:, 1:] * 448	# scale bounding boxes
# 		print(boxes.shape)
# 		fig, ax = plt.subplots()
# 		ax.imshow(x)
# 		for box in boxes:
# 			print(box.shape)
# 			rect = patches.Rectangle((box[1], box[2]), width=box[3], height=-box[4], linewidth=1, edgecolor='r', facecolor='none')
# 			ax.add_patch(rect)
# 		plt.show()	
# 		input()