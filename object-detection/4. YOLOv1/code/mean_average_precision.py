import tensorflow as tf
import numpy as np
from collections import Counter
from loss import YOLOLoss


# TODO: implement in TF completely and should be vectorized


def mAP(true_boxes, pred_boxes, iou_threshold=0.5, classes=20):
	"""
	Computes Mean Average Precision
	
	Parameters:
	true_boxes: a list containing ground truth boxes of the form 
				[image_index, class, confidence, box_x, box_y, box_w, box_h]
	pred_boxes: a list containing predicted boxes of the form 
				[image_index, class, confidence, box_x, box_y, box_w, box_h]
	iou_threshold: IOU threshold for deciding True Positive and False Positive
	classes: No. of classes
	"""

	print("len(true_boxes)", len(true_boxes))
	print("len(pred_boxes)", len(pred_boxes))

	average_precisions = []
	epsilon = []

	for class_ in range(classes):
		print(f"Class {class_}")
		true_class_boxes = []
		pred_class_boxes = []

		# collect all predicted boxes having class "class_"
		for pred_box in pred_boxes:
			if pred_box[1] == class_:
				pred_class_boxes.append(pred_box)

		# collect all true boxes having class "class_"
		for true_box in true_boxes:
			if true_box[1] == class_:
				true_class_boxes.append(true_box)

		# find count of true boxes in each image
		no_of_boxes = Counter( [true_box[0] for true_box in true_class_boxes] )

		# maintain 0/1 for each true box 
		# this is to keep track of the ground truth boxes that are already covered by any predicted box
		for key, count in no_of_boxes.items():
			no_of_boxes[key] = np.zeros(count)


		# sort predicted boxes according to the confidence score
		pred_class_boxes.sort(key=lambda x: x[2], reverse=True)

		# keep running values of true positive and false positive
		TP = np.zeros(len(pred_class_boxes))
		FP = np.zeros(len(pred_class_boxes))

		# for each predicted bounding box, check if it falls in TP or FP
		print("pred_class_boxes:", len(pred_class_boxes))
		for pred_index, pred_box in enumerate(pred_class_boxes):
			
			# find all the true boxes in the same image
			true_image_boxes = [box for box in true_class_boxes if box[0] == pred_box[0]]
			print('true_image_boxes:', true_image_boxes)

			best_iou = 0
			best_iou_index = 0

			# now go through all true boxes in the image and find the one having max iou with 
			# the predicted box
			for true_index, true_box in enumerate(true_image_boxes):
				iou = YOLOLoss.iou(np.expand_dims(true_box[3:], axis=0), np.expand_dims(pred_box[3:], axis=0))

				if iou > best_iou:
					best_iou = iou
					best_iou_index = true_index

			# if this box satisfies the iou_threshold criteria and is not already covered,
			# the predicted is TP
			if best_iou > iou_threshold:
				if no_of_boxes[pred_box[0]][best_iou_index] == 0:
					no_of_boxes[pred_box[0]][best_iou_index] = 1
					TP[pred_index] = 1
				else:
					FP[pred_index] = 1
			else:
				FP[pred_index] = 1

		# in object detection, you maintain running/cumulative sum for precisions and recalls
		TP_cumsum = np.cumsum(TP, axis=0)
		FP_cumsum = np.cumsum(FP, axis=0)


		# recall = TP/TP+FN 
		# But the basic definition of recall is:
		# (no. of correctly predicted positive samples / no. of samples that are actually positive)
		# since all ground truth boxes are to be predicted, the denominator becomes len(true_class_boxes)
		recalls = TP_cumsum / (len(true_class_boxes) + 1e-6)		# to avoid division by 0
		precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

		# the initial values for precision and recall will be 1 and 0 respectively
		precisions = [1] + precisions
		recalls = [0] + recalls
		print("precisions:", precisions)
		print("recalls:", recalls)

		# append the precision for this class_
		average_precisions.append(np.trapz(precisions, recalls))


	# return the Average Precision for the given iou_threshold
	return sum(average_precisions) / len(average_precisions)

