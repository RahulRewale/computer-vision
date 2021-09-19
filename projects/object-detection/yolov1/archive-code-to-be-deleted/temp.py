import numpy as np
import tensorflow.keras as K
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)



def plot_boxes(S, img, label):
	plt.imshow(img)
	ax = plt.gca()
	cell_width, cell_height = img.shape[1]//S, img.shape[0]//S

	# https://stackoverflow.com/a/56180359/6764989
	ax.xaxis.set_major_locator(MultipleLocator(cell_width))
	ax.yaxis.set_major_locator(MultipleLocator(cell_height))
	plt.grid('true')

	for i in range(S):
		for j in range(S):
			if label[i, j, 20] == 1:	# if object exists, plot the ground-truth box
				box = label[i, j, 21:]

				width = box[2] * cell_width
				height = box[3] * cell_height

				mid_x_coord = i*cell_width + cell_width*box[0]
				mid_y_coord = j*cell_height + cell_height*box[1]
				corner_x_coord = mid_x_coord - width/2
				corner_y_coord = mid_y_coord - height/2
				
				rect = patches.Rectangle((corner_x_coord, corner_y_coord), width=width, height=height,
							linewidth=1, edgecolor='r', facecolor='none')
				ax.add_patch(rect)

	plt.show()



def load_label(label_path, S, classes, orig_shape, target_shape):
	print(target_shape)	# height first and then width
	boxes = np.loadtxt(label_path, ndmin=2)

	label = np.zeros((S, S, classes+5), dtype=np.float32)

	cell_height, cell_width = target_shape[0]//S, target_shape[1]//S
	print("cell_height:", cell_height)
	print("cell_width:", cell_width)

	for box in boxes:
		print("original box:", box)	# x, y, width, height
		
		# x-coord and y-coord in the image
		box_img_mid_x = (box[1] * orig_shape[1]) * (target_shape[1]/orig_shape[1])	# x-coord * image width
		box_img_mid_y = (box[2] * orig_shape[0]) * (target_shape[0]/orig_shape[0])	# y-coord * image height

		print('box_img_mid_x', box_img_mid_x)
		print('box_img_mid_y', box_img_mid_y)

		# find the cell the box belongs to
		cell_x = int(box_img_mid_x/cell_width)
		cell_y = int(box_img_mid_y/cell_height)
		print("Cell:", (cell_x, cell_y))

		box_mid_cell_x = (box_img_mid_x - cell_width*cell_x)/cell_width
		box_mid_cell_y = (box_img_mid_y - cell_height*cell_y)/cell_height

		print("box_mid_cell_x", box_mid_cell_x)
		print("box_mid_cell_y", box_mid_cell_y)

		obj_width = (box[3]*orig_shape[1]) * (target_shape[1]/orig_shape[1])
		obj_height = (box[4]*orig_shape[0]) * (target_shape[0]/orig_shape[0])
		print("Object width:", obj_width)
		print("Object height:", obj_height)
		box_width_cell = (obj_width)/cell_width
		box_height_cell = (obj_height)/cell_height

		print("box_width_cell:", box_width_cell)
		print("box_height_cell:", box_height_cell)

		# https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt
		# 0 aeroplane; 1 bicycle; 2 bird; 3 boat; 4 bottle; 5 bus; 6 car; 7 cat; 8 chair; 9 cow; 
		# 10 diningtable; 11 dog; 12 horse; 13 motorbike; 14 person; 15 pottedplant; 16 sheep
		# 17 sofa; 18 train; 19 tvmonitor

		label[cell_x, cell_y, int(box[0])] = 1
		label[cell_x, cell_y, 20:25] = [1, box_mid_cell_x, box_mid_cell_y, box_width_cell, box_height_cell]

	return label


def load_img(img_path, target_shape):
	img = K.preprocessing.image.load_img(img_path)
	img_arr = K.preprocessing.image.img_to_array(img)
	print(img_arr.shape)	# shape gives height first and then width
	img_arr /= 255.0
	return img_arr


target_shape = (448, 448, 3)
img  = load_img(".\\voc_dataset\\images\\000035.jpg", target_shape[:2])
orig_shape = img.shape

boxes = load_label(".\\voc_dataset\\labels\\000035.txt", 9, 20, orig_shape, orig_shape)
plot_boxes(9, img, boxes)

new_img = tf.image.resize(img, target_shape[:2])
print("\n********************\n")
boxes = load_label(".\\voc_dataset\\labels\\000035.txt", 9, 20, target_shape, target_shape)
plot_boxes(9, new_img, boxes)

boxes = load_label(".\\voc_dataset\\labels\\000035.txt", 9, 20, orig_shape, target_shape)
plot_boxes(9, new_img, boxes)