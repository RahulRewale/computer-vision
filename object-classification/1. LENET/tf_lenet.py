'''
This file contains LeNet-5 implementation in tensorflow with a few modifications 
as per recent developments.
'''

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl
import numpy as np
import matplotlib.pyplot as plt


# load data
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

# check data dimensions
# print(x_train.shape)	# (60000, 28, 28)
# print(y_train.shape)	# (60000, )
# print(x_test.shape)	# (10000, 28, 28)
# print(y_test.shape)	# (10000, )
# print(x_train.max())	# 255
# print(x_train.min())	# 0


# normalize data to be in range [0, 1]
x_train = x_train/255.0
x_train = x_train[..., np.newaxis]
x_test = x_test/255.0
x_test = x_test[..., np.newaxis]


# use last 5000 examples for validation
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]


# original LeNet-5 model used tanh activation function, but here we are using 'relu' because it gives better
# results and is the current trend
# LeNet-5 didn't have any dropouts
# LeNet-5 didn't have softmax; it had only a single neuron in the output
model = tf.keras.models.Sequential([
			tfl.ZeroPadding2D(padding=(2, 2), input_shape=x_train[0].shape),	# convert images to (32, 32, 1)
			
			tfl.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh'),						
			tfl.AveragePooling2D(pool_size=(2,2)),

			tfl.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh'),
			tfl.AveragePooling2D(pool_size=(2,2)),

			tfl.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation='tanh'),

			tfl.Flatten(),
			tfl.Dense(units=84, activation='tanh'),
			tfl.Dense(units=10, activation='softmax')
		])

learnRate = 0.009
model.compile(loss = K.losses.SparseCategoricalCrossentropy(from_logits=False),
			optimizer = K.optimizers.SGD(learning_rate=learnRate), metrics = ['accuracy'])

# adding early stopping and, at the end, restore the weights that produced best results on the validation data
print(model.summary())
earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val), callbacks=[earlyStop])

print("*"*15, "Testing", "*"*15)
print(model.evaluate(x_test, y_test))


# Now, predict
pred = model.predict(x_test)

for i in range(15):
	plt.subplot(3, 5, i+1)
	plt.imshow(x_test[i])
	plt.axis('off')
	plt.title("Pred:" + str(np.argmax(pred[i])) + " | Exp:" + str(y_test[i]))

plt.show()

model.save(f'models\\SGD-TANH-LR-{learnRate}')



##### Few Results #####

# RELU - ADAM
# 860/860 [==============================] - 32s 38ms/step - loss: 0.0126 - accuracy: 0.9959 - val_loss: 0.0332 - val_accuracy: 0.9920
# *************** Testing ***************
# 313/313 [==============================] - 3s 8ms/step - loss: 0.0326 - accuracy: 0.9895
# [0.03263542428612709, 0.9894999861717224]


# RELU - SGD - 0.006
# 860/860 [==============================] - 30s 35ms/step - loss: 0.0778 - accuracy: 0.9758 - val_loss: 0.0778 - val_accuracy: 0.9778
# *************** Testing ***************
# 313/313 [==============================] - 2s 7ms/step - loss: 0.0844 - accuracy: 0.9719
# [0.08437399566173553, 0.9718999862670898]


# RELU - SGD - 0.009
# 860/860 [==============================] - 34s 39ms/step - loss: 0.0542 - accuracy: 0.9834 - val_loss: 0.0636 - val_accuracy: 0.9826
# *************** Testing ***************
# 313/313 [==============================] - 3s 10ms/step - loss: 0.0611 - accuracy: 0.9794
# [0.06106874346733093, 0.9793999791145325]


# tanh - ADAM
# 860/860 [==============================] - 31s 37ms/step - loss: 0.0268 - accuracy: 0.9913 - val_loss: 0.0483 - val_accuracy: 0.9868
# *************** Testing ***************
# 313/313 [==============================] - 2s 8ms/step - loss: 0.0542 - accuracy: 0.9823
# [0.05417145416140556, 0.9822999835014343]

# tanh - SGD - 0.009
# 860/860 [==============================] - 37s 43ms/step - loss: 0.0766 - accuracy: 0.9777 - val_loss: 0.0673 - val_accuracy: 0.9806
# *************** Testing ***************
# 313/313 [==============================] - 4s 13ms/step - loss: 0.0729 - accuracy: 0.9772
# [0.07291116565465927, 0.9771999716758728]
