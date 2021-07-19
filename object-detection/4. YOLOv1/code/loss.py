import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class YOLOLoss(K.losses.Loss):
	""" Class used to compute YOLO loss"""
	
	def __init__(self, lambda_coord=5, lambda_noobj=0.5, name='yolo_loss'):
		""" Set few parameters """
		super().__init__(name=name)
		self.lambda_coord = lambda_coord
		self.lambda_noobj = lambda_noobj


	def call(self, y_true, y_pred):
		""" Compute loss 
		
		Parameters:
		y_true: batch containing the ground-truth labels
		y_pred: batch containing the predicted labels
		"""

		### no-object loss: loss when there is no object in a cell ###
		# y_true[..., 20]): the value at index 20 specifies if there is an object or not
		noobj_loss = (1-y_true[..., 20]) * (
							tf.square(y_true[..., 20] - y_pred[..., 20]) +
							tf.square(y_true[..., 20] - y_pred[..., 25])
						)
		# multiply this loss by a factor to reduce its overall weightage in the total loss
		noobj_loss = self.lambda_noobj * tf.reduce_sum(noobj_loss)
		tf.debugging.assert_all_finite(noobj_loss, message='noobj_loss contains nan')


		# Since we predict multiple boxes per grid cell, find the box in each grid cell which is 
		# actually responsible for the prediction using IOU
		iou1 = self.iou(y_true[..., 21:25], y_pred[..., 21:25])
		# tf.debugging.assert_all_finite(iou1, message='iou1 contains nan')
		iou2 = self.iou(y_true[..., 21:25], y_pred[..., 26:30])
		# tf.debugging.assert_all_finite(iou2, message='iou2 contains nan')
		# tf.debugging.assert_shapes([(iou1, (None,7,7)), (iou2, (None, 7,7))])

		# box with max IOU; this is responsible for detecting the object	
		iou_max_ind = tf.math.argmax([iou1, iou2])	# (2,None,7,7)
		iou_max_ind = tf.cast(iou_max_ind, tf.float32)
		# tf.debugging.assert_all_finite(iou_max_ind, message='iou_max_ind contains nan')
		# tf.debugging.assert_shapes([(iou_max_ind, (None, 7,7)), (iou2, (None, 7,7))])

		### object loss - only when there is an object in the cell ###
		obj_loss = y_true[..., 20] * (
									(1-iou_max_ind) * (tf.square(y_true[..., 20] - y_pred[..., 20])) +
									iou_max_ind * (tf.square(y_true[..., 20] - y_pred[..., 25]))
								)
		obj_loss = tf.reduce_sum(obj_loss)
		# tf.debugging.assert_all_finite(obj_loss, message='obj_loss contains nan')


		### bounding box loss - only when there is an object in the cell ###
		# loss due to box 0 coordinates if it is responsible for detection
		coord_loss_box0 = (1-iou_max_ind) * (
							tf.square(y_true[..., 21] - y_pred[..., 21]) +
							tf.square(y_true[..., 22] - y_pred[..., 22])
						)
		# tf.debugging.assert_all_finite(coord_loss_box0, message='coord_loss_box0 contains nan')

		# loss due to box 1 coordinates if it is responsible for detection
		coord_loss_box1 = iou_max_ind * (
							tf.square(y_true[..., 21] - y_pred[..., 26]) + 
							tf.square(y_true[..., 22] - y_pred[..., 27])
						)
		# tf.debugging.assert_all_finite(coord_loss_box1, message='coord_loss_box1 contains nan')

		# only if there is an object in the cell
		coord_loss = y_true[..., 20] * (coord_loss_box0 + coord_loss_box1)
		# Total loss due to coordinates scaled by a some factor
		coord_loss = self.lambda_coord * tf.reduce_sum(coord_loss)


		# loss due to box 0 dimensions if it is responsible for detection
		width_height_loss_box0 = (1-iou_max_ind) * (
								tf.square(tf.math.sqrt(y_true[..., 23]) - 
									tf.math.sign(y_pred[..., 23]) * tf.math.sqrt(tf.math.abs(y_pred[..., 23]) + 1e-6)) +
								tf.square(tf.math.sqrt(y_true[..., 24]) - 
									tf.math.sign(y_pred[..., 24]) * tf.math.sqrt(tf.math.abs(y_pred[..., 24]) + 1e-6))
							)
		# tf.debugging.assert_all_finite(width_height_loss_box0, message='width_height_loss_box0 contains nan')

		# loss due to box 1 dimensions if it is responsible for detection
		width_height_loss_box1 = iou_max_ind * (
								tf.square(tf.math.sqrt(y_true[..., 23]) - 
									tf.math.sign(y_pred[..., 28]) * tf.math.sqrt(tf.math.abs(y_pred[..., 28]) + 1e-6)) +
								tf.square(tf.math.sqrt(y_true[..., 24]) - 
									tf.math.sign(y_pred[..., 29]) * tf.math.sqrt(tf.math.abs(y_pred[..., 29]) + 1e-6))
							)
		# tf.debugging.assert_all_finite(width_height_loss_box1, message='width_height_loss_box1 contains nan')

		# only if there is an object in the cell
		width_height_loss = y_true[..., 20] * (width_height_loss_box0 + width_height_loss_box1)
		width_height_loss = self.lambda_coord * tf.reduce_sum(width_height_loss)


		### classification loss ###
		class_loss = y_true[..., 20:21] * tf.square(y_true[..., :20] - y_pred[..., :20])
		class_loss = tf.reduce_sum(class_loss)

		return noobj_loss + obj_loss + class_loss + coord_loss + width_height_loss
		

	@staticmethod
	def iou(true_box, pred_box):
		"""Computes IOU between two batches of bounding boxes"""
		
		# obtain corner coordinates for predicted box
		pred_box_x1 = pred_box[...,0] - pred_box[...,2]/2
		pred_box_x2 = pred_box[...,0] + pred_box[...,2]/2
		pred_box_y1 = pred_box[...,1] - pred_box[...,3]/2
		pred_box_y2 = pred_box[...,1] + pred_box[...,3]/2
		# tf.debugging.assert_shapes([(pred_box_x1, (None, 7, 7)), (pred_box_y1, (None, 7, 7))])

		# obtain corner coordinates for true box
		true_box_x1 = true_box[...,0] - true_box[...,2]/2
		true_box_x2 = true_box[...,0] + true_box[...,2]/2
		true_box_y1 = true_box[...,1] - true_box[...,3]/2
		true_box_y2 = true_box[...,1] + true_box[...,3]/2
		# tf.debugging.assert_shapes([(true_box_x1, (None, 7, 7)), (true_box_y1, (None, 7, 7))])

		# obtain corners of the intersection
		inter_x1 = tf.math.maximum(pred_box_x1, true_box_x1)
		inter_y1 = tf.math.maximum(pred_box_y1, true_box_y1)
		inter_x2 = tf.math.minimum(pred_box_x2, true_box_x2)
		inter_y2 = tf.math.minimum(pred_box_y2, true_box_y2)
		# tf.debugging.assert_shapes([(inter_x1, (None, 7, 7)), (inter_y1, (None, 7, 7))])	

		# intersection area
		# inter_x2-inter_x1 and/or inter_y2 - inter_y1 < 0, the boxes don't intersect
		inter_area = tf.math.maximum((inter_x2 - inter_x1), 0) * tf.math.maximum((inter_y2 - inter_y1), 0)
		# tf.debugging.assert_shapes([(inter_area, (None, 7, 7))])

		# union area
		union_area = tf.math.abs((pred_box_x2 - pred_box_x1) * (pred_box_y2 - pred_box_y1)) + \
						tf.math.abs((true_box_x2 - true_box_x1) * (true_box_y2 - true_box_y1)) - \
						inter_area
		# tf.debugging.assert_shapes([(inter_area, (None, 7, 7))])

		return inter_area/(union_area + 1e-6)