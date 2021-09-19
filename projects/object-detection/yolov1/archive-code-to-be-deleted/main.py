import loss
import load_data
import model
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import sys
import numpy as np


def plot_boxes(S, img, label, pred):
	plt.imshow(img)
	ax = plt.gca()
	cell_width, cell_height = img.shape[1]//S, img.shape[0]//S

	# https://stackoverflow.com/a/56180359/6764989
	# ax.xaxis.set_major_locator(MultipleLocator(cell_width))
	# ax.yaxis.set_major_locator(MultipleLocator(cell_height))
	# plt.grid('true')

	iou1 = loss.iou(np.expand_dims(label[..., 21:], axis=0), np.expand_dims(pred[..., 21:25], axis=0))[0]
	iou2 = loss.iou(np.expand_dims(label[..., 21:], axis=0), np.expand_dims(pred[..., 26:30], axis=0))[0]

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

	plt.imshow(img)
	ax = plt.gca()
	for i in range(S):
		for j in range(S):
			if label[i, j, 20] == 1:	# if object exists, plot the ground-truth box
				# find the predicted box with highest iou and plot it
				ind = np.argmax([iou1[i, j], iou2[i,j]])
				if ind == 0:
					pred_box = pred[i, j, 21:25]
				else:
					pred_box = pred[i, j, 26:30]
				
				width = pred_box[2] * cell_width
				height = pred_box[3] * cell_height

				mid_x_coord = i*cell_width + cell_width*pred_box[0]
				mid_y_coord = j*cell_height + cell_height*pred_box[1]
				
				corner_x_coord = mid_x_coord - width/2
				corner_y_coord = mid_y_coord - height/2
				
				rect = patches.Rectangle((corner_x_coord, corner_y_coord), width=width, height=height,
							linewidth=1, edgecolor='b', facecolor='none')
				ax.add_patch(rect)

	plt.show()




GRID_SIZE=7
BOXES=2
CLASSES=20
TARGET_SHAPE = (448, 448, 3)

ds = load_data.load_train_data('.\\voc_dataset\\8examples.csv', S=GRID_SIZE, C=CLASSES, target_shape=TARGET_SHAPE)
ds = ds.batch(16)

# verify images and bounding boxes are loaded correctly
# for img_batch, label_batch in ds.take(1):
# 	for img, label in zip(img_batch, label_batch):
# 		print(img.shape)
# 		print(label.shape)
# 		plot_boxes(GRID_SIZE, img.numpy(), label.numpy())


model = model.create_model(TARGET_SHAPE, S=GRID_SIZE, B=BOXES, classes=CLASSES)

# check if model works
# for img_batch, label_batch in ds.take(1):
# 		print(model(img_batch).numpy().shape)


# optimizer_fn= K.optimizers.SGD(learning_rate=0.05)
# without momentum: 0.00005
learnRate = float(sys.argv[1])
epochs = int(sys.argv[2])
model.compile(optimizer=K.optimizers.SGD(learning_rate=learnRate, momentum=0.9), 
				loss=loss.loss_fn)

reduceLR = K.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, mode='min',
				min_delta=5, min_lr=0.000001)
# ckpt = K.callbacks.ModelCheckpoint("trained-models\\{epoch:02d}-{loss:.5f}", 
# 			monitor='loss', save_weights_only=True, save_best_only=True)

model.fit(ds, epochs=epochs)


for img_batch, label_batch in ds.take(1):
	preds_batch = model.predict(img_batch)
	for img, label, pred in zip(img_batch, label_batch, preds_batch):
		plot_boxes(GRID_SIZE, img.numpy(), label.numpy(), pred)

model.save(f'trained-model-{learnRate}-{epochs}')

# # train with a smaller learning rate
# model.compile(optimizer=K.optimizers.SGD(learning_rate=learnRate/5),
# 				loss=loss.loss_fn)
# model.fit(ds.take(10), epochs=epochs-25)


# for img_batch, label_batch in ds.take(1):
# 	preds_batch = model.predict(img_batch)
# 	for img, label, pred in zip(img_batch, label_batch, preds_batch):
# 		plot_boxes(GRID_SIZE, img.numpy(), label.numpy(), pred)

