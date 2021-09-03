"""
	This file creates an alexnet model, trains it, and then evaluates it.
	You can pass below parameters to this script:
		learning rate
		epochs 
		no. of units in the dense layers
		dropout rate
		weight decay factor
"""


from  data_loader import DataLoader
from alexnet import AlexNet
import tensorflow.keras as K
import sys, os
import numpy as np
import pandas as pd


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
	dense_units = 64
	drop = 0.5
	weight_decay = 0.015


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
ds_loader = DataLoader("..\\datasets\\augmented",
							"..\\datasets\\imagenette2-320\\val",
							"..\\datasets\\imagenette2-320\\val")
train_ds = ds_loader.load_train_data(input_shape[:2])
val_ds, test_ds = ds_loader.load_val_test_data(input_shape[:2])

# create model
alex_net = AlexNet(dense_units=dense_units, drop=drop, weight_decay=weight_decay)
K.utils.plot_model(alex_net, 'mymodel.png', show_shapes=True)

#configure model
optimizer = K.optimizers.SGD(learning_rate=lr, momentum=0.9)
alex_net.compile(optimizer=optimizer, loss=K.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# callbacks
tensorBoard = K.callbacks.TensorBoard(log_dir=tbLogsPath)
# earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
# Reduce learning rate if validation loss doesn't improve much for few consecutive epochs
reduceLR = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.05, 
										patience=3, cooldown=1, min_lr=0.00001, verbose=1)
ckpt = K.callbacks.ModelCheckpoint(
			checkPointPath + 'checkpoint-epoch{epoch:02d}-loss{val_loss:.3f}',
			save_weights_only=True, monitor='val_accuracy', 
			mode='max', save_best_only=True)
callbacks = [tensorBoard, reduceLR, ckpt]

# train model
history = alex_net.fit(x=train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

# save model training history to a csv file
df = pd.DataFrame(history.history)
df.to_csv(checkPointPath + 'history.csv')

# alex_net.save(f"{lr}-{epochs}")

# test the model
print(alex_net.evaluate(test_ds))

# plot few samples from test_ds
batchX, batchY = next(test_ds.as_numpy_iterator())
pred = alex_net.predict(batchX)

for i in range(15):	# plot few images from the batch
	plt.subplot(3, 5, i+1)
	plt.imshow(batchX[i])
	plt.title(classes[np.argmax(pred[i])])
	plt.axis('off')

plt.tight_layout()
# save the figure
plt.savefig(checkPointPath + 'test-sample.png')