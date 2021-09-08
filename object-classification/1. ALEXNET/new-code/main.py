"""
	This file creates an alexnet model, trains it, and then evaluates it.
	You can pass below parameters to this script:
		learning rate
		epochs 
		no. of units in the dense layers
		dropout rate
		weight decay factor
"""


from  new_data_loader import DataLoader
from alexnet import AlexNet
import tensorflow as tf
import tensorflow.keras as K
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


assert len(sys.argv) > 2, f"Pass learning rate and epochs at least; \
							in addition you can pass no. of units in dense layers (default 512), \
							dropout rate (default 0.6), and weight decay (default 0.008)"

lr = float(sys.argv[1])				# 0.008
epochs = int(sys.argv[2])			# 20
input_shape = (227, 227, 3)

if len(sys.argv) > 3:
	dense_units = int(sys.argv[2])
	drop = float(sys.argv[3])
	weight_decay = float(sys.argv[4])
else:		# default values
	dense_units = 32
	drop = 0.5
	weight_decay = 0.05


print("*"*10, f"LR-{lr}-epochs-{epochs}-units-{dense_units}-drop-{drop}-wd-{weight_decay}", "*"*10)


# create directories for storing checkpoints and tensorflow graph
logPath = f'LR-{lr}'
checkPointPath = f'.\\checkpoints\\{logPath}\\'
tbLogsPath = f'.\\tb-logs\\{logPath}'
os.makedirs(checkPointPath, exist_ok=True)
os.makedirs(tbLogsPath, exist_ok=True)


# Object Classes: https://github.com/fastai/imagenette#imagenette-1
classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']


# load all data
ds_loader = DataLoader("..\\..\\datasets\\imagenette2-320\\train",
							"..\\..\\datasets\\imagenette2-320\\val",
							"..\\..\\datasets\\imagenette2-320\\val")
train_ds = ds_loader.load_train_data()
val_ds, test_ds = ds_loader.load_val_test_data()

# each batch in val_ds and test_ds has shape (?, 10, 227, 227, 3) for x and (?, 10) for y

# create model
alex_net = AlexNet(dense_units=dense_units, drop=drop, weight_decay=weight_decay)
K.utils.plot_model(alex_net, 'mymodel.png', show_shapes=True)

#configure model
optimizer = K.optimizers.SGD(learning_rate=lr, momentum=0.9)
# alex_net.compile(optimizer=optimizer, loss=K.losses.CategoricalCrossentropy(), metrics=['accuracy'])

loss_obj = K.losses.CategoricalCrossentropy()

train_mean_loss = K.metrics.Mean()
train_cat_acc = K.metrics.CategoricalAccuracy()

mean_loss = K.metrics.Mean()
cat_acc = K.metrics.CategoricalAccuracy()

losses = []
val_losses = []
accuracies = []
val_accuracies = []

@tf.function
def train_step(batch_x, batch_y):
	with tf.GradientTape() as tape:
		pred = alex_net(batch_x, training=True)
		loss = loss_obj(batch_y, pred)
	grads = tape.gradient(loss, alex_net.trainable_variables)
	optimizer.apply_gradients(zip(grads, alex_net.trainable_variables))
	train_mean_loss(loss)
	train_cat_acc.update_state(batch_y, pred)


def compute_metrics(ds):
	cat_acc.reset_states()
	mean_loss.reset_states()
	for batch_x, batch_y in ds:
		# do croppping and averaging outputs here
		pred = alex_net(tf.reshape(batch_x, (-1, 227, 227, 3)), training=False)
		pred = tf.reshape(pred, (-1, 10, 10))
		pred = tf.reduce_mean(pred, axis=1)
		loss = loss_obj(batch_y, pred)
		mean_loss(loss)
		cat_acc.update_state(batch_y, pred)


no_of_train_batches = train_ds.cardinality()
print(f"Training dataset size: {no_of_train_batches}")

# min_val_acc_delta = 0.05
for epoch in range(epochs):
	print(f"\n\nEpoch {epoch+1}/{epochs}:")

	print(f"Learning rate: {optimizer.learning_rate}")
	
	train_mean_loss.reset_states()
	train_cat_acc.reset_states()

	batch_no = 0
	for batch_x, batch_y in train_ds:
		batch_no += 1
		train_step(batch_x, batch_y)
		if batch_no%100 == 0:
			print(f"\tLoss for {batch_no}/{no_of_train_batches}: {train_mean_loss.result()}")

	no_of_train_batches = batch_no
	print(f"\nTraining Metrics: Loss {train_mean_loss.result()}; Accuracy {train_cat_acc.result()}")
	losses.append(train_mean_loss.result().numpy())
	accuracies.append(train_cat_acc.result().numpy())

	compute_metrics(val_ds)
	print(f"Validation Metrics: Loss {mean_loss.result()}; Accuracy {cat_acc.result()}")
	val_losses.append(mean_loss.result().numpy())
	val_accuracies.append(cat_acc.result().numpy())

	# epoch is not zero and epoch is even
	if epoch and epoch%2==0:
		optimizer.learning_rate = optimizer.learning_rate * 0.8



# test the model
compute_metrics(test_ds)
print(f"\n\nTesting Metrics: Loss {mean_loss.result()}; Accuracy {cat_acc.result()}")

losses.append(0)
accuracies.append(0)
val_losses.append(mean_loss.result().numpy())
val_accuracies.append(cat_acc.result().numpy())

# save model training history to a csv file
df = pd.DataFrame(np.vstack((losses, val_losses, accuracies, val_accuracies)).T, columns=["loss", "val_loss", "accuracy", "val_accuracy"])
df.to_csv(checkPointPath + 'history.csv')

# alex_net.save(f"{lr}-{epochs}")

# plot few samples from test_ds
batch_x, batch_y = next(test_ds.as_numpy_iterator())
pred = alex_net(tf.reshape(batch_x, (-1, 227, 227, 3)), training=False)
pred = tf.reshape(pred, (-1, 10, 10))
pred = tf.reduce_mean(pred, axis=1).numpy()

for i in range(15):	# plot few images from the batch
	plt.subplot(3, 5, i+1)
	plt.imshow(batch_x[i,4])	# take the central crop (4th) for displaying
	plt.title(classes[np.argmax(pred[i])])
	plt.axis('off')

plt.tight_layout()
# save the figure
plt.savefig(checkPointPath + 'test-sample.png')