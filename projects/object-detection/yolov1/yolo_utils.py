import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
from loss import YOLOLoss
import mean_average_precision
import sys


# classes of objects that we are trying to detect
classes = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car',
		 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 
		 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}


def plot_boxes(grid_size, img, label, pred):
	"""
	Plots two subplots - one showing the given image along with the ground truth boxes and the other
						 showing the given image along with the predicted boxes

	Parameters:
	grid_size: Grid size
	img: img to display
	label: ground truth boxes for the image
	pred: predicted boxes for the image
	"""

	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.imshow(img)
	ax2.imshow(img)

	cell_width, cell_height = img.shape[1]/grid_size, img.shape[0]/grid_size

	# to see grid cells, uncomment below code
	# https://stackoverflow.com/a/56180359/6764989
	# ax1.xaxis.set_major_locator(MultipleLocator(cell_width))
	# ax1.yaxis.set_major_locator(MultipleLocator(cell_height))
	# ax1.grid('true')
	# ax2.xaxis.set_major_locator(MultipleLocator(cell_width))
	# ax2.yaxis.set_major_locator(MultipleLocator(cell_height))
	# ax2.grid('true')

	# compute IOU for both predicted boxes per cell
	iou1 = YOLOLoss.iou(np.expand_dims(label[..., 21:], axis=0), np.expand_dims(pred[..., 21:25], axis=0))[0]
	iou2 = YOLOLoss.iou(np.expand_dims(label[..., 21:], axis=0), np.expand_dims(pred[..., 26:30], axis=0))[0]

	for i in range(grid_size):
		for j in range(grid_size):
			if label[i, j, 20] == 1:	# if object exists, plot the ground-truth box

				img_box = cell_box_to_actual_box(img.shape, grid_size, i, j, label[i, j, 21:])
				rect = patches.Rectangle((img_box[0], img_box[1]), width=img_box[4], height=img_box[5],
							linewidth=1, edgecolor='r', facecolor='none', label=classes[np.argmax(label[i, j, 0:20])])
				ax1.add_patch(rect)
				# plot object midpoint as well
				ax1.plot(img_box[2], img_box[3],'ro')
				ax1.legend()

				# find the predicted box with the highest iou and plot it
				ind = np.argmax([iou1[i, j], iou2[i,j]])
				pred_box = pred[i, j, 21+ind*5:25+ind*5]
				
				img_box = cell_box_to_actual_box(img.shape, grid_size, i, j, pred_box)
				rect = patches.Rectangle((img_box[0], img_box[1]), width=img_box[4], height=img_box[5],
							linewidth=1, edgecolor='b', facecolor='none', label=classes[np.argmax(pred[i, j, 0:20])])
				ax2.add_patch(rect)
				# plot object midpoint as well
				ax2.plot(img_box[2], img_box[3],'bo')
				ax2.legend()

	plt.show()


def cell_box_to_actual_box(img_shape, grid_size, i, j, cell_box):
	"""
	Converts bounding box that is measured relative to the cell to the actual size
	
	Parameters:
	img_shape: shape of the image (i.e. actual size)
	grid_size: grid size used
	i: x-coordinate of the cell to which the box belongs
	j: y-coordinate of the cell to which the box belongs
	cell_box: box measured relative to the cell; of the form [x, y, w, h]
	"""

	cell_width, cell_height = img_shape[1]/grid_size, img_shape[0]/grid_size
	width = cell_box[2] * cell_width
	height = cell_box[3] * cell_height

	mid_x_coord = i*cell_width + cell_width*cell_box[0]
	mid_y_coord = j*cell_height + cell_height*cell_box[1]

	corner_x_coord = mid_x_coord - width/2
	corner_y_coord = mid_y_coord - height/2
	return (corner_x_coord, corner_y_coord, mid_x_coord, mid_y_coord, width, height)


# tested - working fine
def cell_boxes_img_boxes(cell_boxes_batch, img_size=448, grid_size=7):
	"""
	Converts a batch of boxes measured relative to the cell dimensions to the boxes measured relative
	to the given image

	Parameters:
	cell_boxes_batch: batch of boxes measured relative to the cell dimensions
	img_size: size of images used
	grid_size: grid size used
	"""

	cell_size = img_size/grid_size

	# multiply x-coord, y-coord, width, and height with the cell size
	cell_boxes_batch[..., :] = cell_boxes_batch[..., :] * cell_size

	# add the cell offsets to the x and y coords
	grid_of_cell_indices = np.array([ [i, j] for j in range(grid_size) for i in range(grid_size)])
	grid_of_cell_indices = grid_of_cell_indices.reshape((grid_size, grid_size, 2))
	cell_boxes_batch[..., 0:2] += grid_of_cell_indices * cell_size

	# boxes relative to the whole image
	cell_boxes_batch[..., :] = cell_boxes_batch[..., :]/img_size

	return cell_boxes_batch


# TODO - not working; migrate to TF
def compute_map(y_true_batch, y_pred_batch, grid_size=7, threshold=0.5, iou_threshold=0.5):
	"""
	Preprocesses ground-truth and predicted boxes and then computes mean average precision

	Parameters:
	y_true_batch: a batch of labels
	y_pred_batch: a batch of predictions
	grid_size: grid size used
	threshold: threshold for object presence probability
	iou_threshold: threshold for IOU
	"""

	np.set_printoptions(threshold=sys.maxsize)

	# tempFile = open('debug.log', mode='at')
	# convert the box coordinates relative to the whole image
	# print("y_true_batch[0:2, 0:2, 0:2 , 21:25]:\n", y_true_batch[0:2, 0:2, 0:2 , 21:25])
	y_true_batch[..., 21:25] = cell_boxes_img_boxes(y_true_batch[..., 21:25], grid_size=grid_size)
	# print("\ny_true_batch[0:2, 0:2, 0:2 , 21:25]:\n", y_true_batch[0:2, 0:2, 0:2 , 21:25])

	# pick class index, object probability, bounding box coordinates
	true_boxes = y_true_batch[..., 20:25]	# (None, S, S, 5)
	true_class = np.expand_dims(np.argmax(y_true_batch[..., 0:20], axis=-1), axis=-1) # (None, S, S, 1)
	true_class_boxes = np.concatenate( (true_class, true_boxes), axis=-1) # (None, S, S, 6)

	# convert the box coordinates relative to the whole image
	y_pred_batch[..., 21:25] = cell_boxes_img_boxes(y_pred_batch[..., 21:25], grid_size=grid_size)
	y_pred_batch[..., 26:30] = cell_boxes_img_boxes(y_pred_batch[..., 26:30], grid_size=grid_size)
	
	# pick one of the two boxes in each cell
	# and then pick class index, object probability, bounding box coordinates
	max_ind = np.argmax([y_pred_batch[..., 20:21], y_pred_batch[..., 25:26]], axis=0) # [(None, S, S, 1), (None, S, S, 1)]
	pred_boxes = (1-max_ind) * y_pred_batch[..., 20:25] + max_ind * y_pred_batch[..., 25:30] # (None, S, S, 5)
	pred_class = np.expand_dims(np.argmax(y_pred_batch[..., 0:20], axis=-1), axis=-1)	# (None, S, S, 1)
	pred_class_boxes = np.concatenate( (pred_class, pred_boxes), axis=-1) # (None, S, S, 6)

	true_class_boxes = true_class_boxes.reshape((-1, grid_size*grid_size, 6))
	pred_class_boxes = pred_class_boxes.reshape((-1, grid_size*grid_size, 6))
	# print("true_class_boxes:\n", true_class_boxes)
	# print("\npred_class_boxes:\n", pred_class_boxes)
	all_true_boxes = []
	all_pred_boxes = []

	# print("\n\n")
	for index in range(y_true_batch.shape[0]):
		print(f"***Image: {index}***")
		true_img_box_list = []
		pred_img_box_list = []

		for box_ind in range(grid_size * grid_size):
			pred_img_box_list.append(pred_class_boxes[index, box_ind, :])

			if true_class_boxes[index, box_ind, 1] > threshold:
				true_img_box_list.append([index, *true_class_boxes[index, box_ind, :]])

		print(f"Total expected boxes: {len(true_img_box_list)}")
		# print(f"Total predicted boxes: {len(pred_img_box_list)}")
		
		pred_img_box_list = non_max_suppression(pred_img_box_list, threshold, iou_threshold)

		for box_ind in range(len(pred_img_box_list)):
			pred_img_box_list[box_ind] = [index, *pred_img_box_list[box_ind]]

		
		all_pred_boxes.extend(pred_img_box_list)
		all_true_boxes.extend(true_img_box_list)

	print("all_true_boxes:", all_true_boxes)
	print("all_pred_boxes:", all_pred_boxes)

	return mean_average_precision.mAP(all_true_boxes, all_pred_boxes, iou_threshold, 20)


# TODO - not working; migrate to TF
def non_max_suppression(given_pred_boxes, threshold, iou_threshold): # for one image at a time
	"""
	Performs Non-max supression on the given predicted boxes in an image

	Parameters:
	given_pred_boxes: Predicted boxes in a single image; 
					  Of the form [class, confidence, box_x, box_y, box_w, box_h]
	threshold: threshold for object presence probability
	iou_threshold: threshold for IOU
	"""

	# keep boxes satisfying minimum confidence criteria
	# print(f"Total predicted boxes: {len(given_pred_boxes)}")
	pred_boxes = [pred_box for pred_box in given_pred_boxes if pred_box[1] > threshold]
	print(f"Total predicted boxes > {threshold}: {len(pred_boxes)}")
	# sort boxes according to probabilities
	pred_boxes = sorted(pred_boxes, key=lambda x: x[1], reverse=True)

	output_boxes = []

	while pred_boxes:
		# pick the box with highest probability
		best_box = pred_boxes.pop(0)
		print("best_box[0]:", best_box[1])

		# keep boxes having different class or same class but iou lesser than threshold 
		filtered_boxes = []
		for box in pred_boxes:
			iou = YOLOLoss.iou(np.expand_dims(best_box[2:], axis=0), np.expand_dims(box[2:], axis=0))
			if (box[0] != best_box[0]) or (iou < iou_threshold):
				filtered_boxes.append(box)

		pred_boxes = filtered_boxes
		output_boxes.append(best_box)	# the box to output

	return output_boxes



if __name__ == '__main__':
	inp = np.array([
					[ [ [0.5,0.5,2,3], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],	# (1/8, 1/8, 4/8, 6/8)
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ] ],

					[ [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0.5,0.5,1.3,3], [0,0,0,0], [0,0,0,0] ], # (3/8, 5/8, 2.6/8, 6/8)
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ] ],

					[ [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ],
					  [ [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] ] ]
				])
	res = cell_boxes_img_boxes(inp, img_size=8, grid_size=4)
	print('\n\n')
	print(res[0])
	print('\n\n')
	print(res[1])