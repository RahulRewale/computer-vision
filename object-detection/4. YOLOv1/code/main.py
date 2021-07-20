import tensorflow as tf
import tensorflow.keras as K
import loss
import sys
from data_loader import DataLoader
from model import YOLOV1
import yolo_utils
import os



def schedule(epoch, lr):
	""" A scheduler for modifying learning rate """
	if 0<epoch<5:
		return lr*1.1
	elif epoch==20:
		return lr/2
	elif epoch==30:
		return lr/2		
	elif epoch==40:
		return lr/2
	elif epoch==45:
		return lr/2
	else:
		return lr


@tf.function
def train_fn(model, img_batch, label_batch):
	""" Function for forward propagation, computing loss, and computing gradients 

	Parameters:
	model: the model to train
	img_batch: batch of input images
	label_batch: batch of labels for img_batch
	"""

	yolo_loss = loss.YOLOLoss()
	with tf.GradientTape() as tape:
		pred_batch = model(img_batch, training=True)
		batch_loss = yolo_loss(label_batch, pred_batch)
	grads = tape.gradient(batch_loss, model.trainable_weights)
	return batch_loss, grads



# make sure learning rate and epochs are given
assert len(sys.argv) > 2, 'Please pass learning rate and epoch count; e.g. python main.py 0.00005 10'

# some constants
GRID_SIZE=7
BOXES=2
CLASSES=20
TARGET_SHAPE = (448, 448, 3)

learnRate = float(sys.argv[1])	# 0.00005
epochs = int(sys.argv[2])


# create a direcotory for storing results
if not os.path.exists(f'trained-model-{learnRate}-{epochs}'):
	os.makedirs(f'trained-model-{learnRate}-{epochs}')


# load training data
loader = DataLoader("voc_dataset\\images", "voc_dataset\\labels")
train_ds = loader.load_train_data('.\\voc_dataset\\100examples.csv', GRID_SIZE, CLASSES, TARGET_SHAPE)
train_ds = train_ds.batch(16)
val_ds = loader.load_train_data('.\\voc_dataset\\8examples.csv', GRID_SIZE, CLASSES, TARGET_SHAPE)
val_ds = val_ds.batch(16)


# create model and train
model = YOLOV1((448, 448, 3), GRID_SIZE, BOXES, CLASSES)
optimizer = K.optimizers.SGD(learning_rate=learnRate, momentum=0.9)

if len(sys.argv) > 3:
	model.load_yolo(sys.argv[3], optimizer)
	print("Loaded pre-trained model and optimizer")


# callbacks
ckpt = K.callbacks.ModelCheckpoint("trained-models\\{epoch:02d}-{loss:.5f}", 
  			monitor='loss', save_weights_only=True, save_best_only=True)
csv_logger = K.callbacks.CSVLogger(f'trained-model-{learnRate}-{epochs}\\training_logs.csv')
lr_scheduler = K.callbacks.LearningRateScheduler(schedule, verbose=1)

# automated training
model.compile(optimizer=optimizer, loss=loss.YOLOLoss())
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[lr_scheduler, csv_logger, ckpt])


# custom training
# metric = K.metrics.Mean()
# for ep in range(epochs):
# 	print(f"\n\nEpoch {ep+1}/{epochs}:")
# 	batch_losses = []
# 	metric.reset_state()

# 	# change learning rate every 5 epochs
# 	if (ep != 0) and (ep % 10 == 0):
# 		print("Learning rate:", optimizer.lr.numpy())
# 		optimizer.lr = optimizer.lr * 0.8
# 		print("New Learning rate:", optimizer.lr.numpy())

# 	for index, (img_batch, label_batch) in enumerate(train_ds):
# 		batch_loss, grads = train_fn(model, img_batch, label_batch)
# 		optimizer.apply_gradients(zip(grads, model.trainable_weights))
# 		batch_losses.append(batch_loss)
 		
#  		# TODO - implement Mean Average Precision (in Tensorflow)
#  		# TODO - below numpy implementation seems to have bug(s)
# 		metric.update_state(yolo_utils.compute_map(label_batch.numpy(), model(img_batch, training=False).numpy(), GRID_SIZE, threshold=0.4, iou_threshold=0.5))

# 	print(f"Loss for epoch {ep+1}:", sum(batch_losses)/len(batch_losses))
# 	print(metric.result())	# print mAP value


# plot few training images to see the result
for img_batch, label_batch in train_ds.take(1):
	preds_batch = model.predict_on_batch(img_batch)
	for img, label, pred in zip(img_batch, label_batch, preds_batch):
		yolo_utils.plot_boxes(GRID_SIZE, img.numpy(), label.numpy(), pred)
		print("\n\n")

# plot few validation images to see the result
for img_batch, label_batch in val_ds.take(1):
	preds_batch = model.predict_on_batch(img_batch)
	for img, label, pred in zip(img_batch, label_batch, preds_batch):
		yolo_utils.plot_boxes(GRID_SIZE, img.numpy(), label.numpy(), pred)

# save the model
model.save_yolo(f'trained-model-{learnRate}-{epochs}', optimizer)
print("Model saved")
