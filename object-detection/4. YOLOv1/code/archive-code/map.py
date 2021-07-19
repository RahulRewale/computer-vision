import tensorflow as tf
import tensorflow.keras as K


def mean_avg_precision(y_true, y_pred, C=20, threshold=0.5):
	gt_boxes = y_true[..., ]
	tf.argmax(y_true[...,])
	for category in range(C):

