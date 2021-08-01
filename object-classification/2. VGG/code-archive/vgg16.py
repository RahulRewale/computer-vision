import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd


def loadData():
	trainDS = K.preprocessing.image_dataset_from_directory(
			directory = '..\\datasets\\augmented',
			label_mode = 'categorical', 
			batch_size = 64, 
			image_size = (227, 227)
			)

	valTestDS = K.preprocessing.image_dataset_from_directory(
				directory = '..\\datasets\\imagenette2-320\\val',
				label_mode = 'categorical',
				batch_size = 64,
				image_size = (227, 227)
				)

	noOfExamples = tf.data.experimental.cardinality(valTestDS).numpy()
	valDS = valTestDS.take(noOfExamples/2)		# first 50%
	testDS = valTestDS.skip(noOfExamples/2)		# remaining 50%

	# since in transfer learning we generally have small amount of data, we return only 
	# a few batches here
	return trainDS.take(50), valDS.take(30), testDS



learnRate  = float(sys.argv[1])
outDir = f'.\\models\\trainedmodel-{learnRate}\\'
os.makedirs(outDir)

classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']


vggModel = K.applications.VGG16(include_top=False, weights='imagenet', input_shape=(227, 227, 3))
vggModel.trainable = False


trainDS, valDS, testDS = loadData()
# caching doesn't work on my system due to small amount of RAM
# trainDS = trainDS.cache().prefetch(tf.data.experimental.AUTOTUNE)
# valDS = valDS.cache().prefetch(tf.data.experimental.AUTOTUNE)
# testDS = testDS.cache().prefetch(tf.data.AUTOTUNE)


inp = K.Input(shape=(227, 227, 3))
x = K.applications.vgg16.preprocess_input(inp)
x = vggModel(x)
x = K.layers.Flatten()(x)
x = K.layers.Dense(units=2048, activation='relu')(x)
x = K.layers.Dropout(0.6) (x)
x = K.layers.Dense(units=2048, activation='relu')(x)
x = K.layers.Dropout(0.6) (x)
out = K.layers.Dense(units=10, activation='softmax')(x)
model = K.Model(inputs=inp, outputs=out)

K.utils.plot_model(model, to_file="model.png", show_shapes=True)

model.compile(loss='categorical_crossentropy', 
				optimizer=K.optimizers.SGD(learning_rate=learnRate),
 				metrics=['accuracy']
 				)


img = K.preprocessing.image.load_img('test-images\\chainsaw.jpeg', target_size=(227, 227))
arr = K.preprocessing.image.img_to_array(img)
arr = arr[np.newaxis, ...]

print('\n\nTesting the pre-trained model')
pred = model.predict(arr)
print("*"*10, pred, "*"*10)
plt.imshow(arr[0]/255.0)
plt.axis('off')
plt.title(classes[np.argmax(pred[0])])
# plt.show()

print(model.evaluate(testDS))


print('\n\nTraining the pre-trained model')
earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(trainDS, epochs=25, validation_data=valDS, callbacks=[earlyStop])

pred = model.predict(arr)
print("*"*10, pred, "*"*10)

print('\n\nTesting the tuned model')
testResults = model.evaluate(testDS)
print(testResults)

plt.imshow(arr[0]/255.0)
plt.axis('off')
plt.title(classes[np.argmax(pred[0])])
plt.show()


df_train = pd.DataFrame(history.history)
df_test = pd.DataFrame([testResults], columns=['test_loss', 'test_acc'])
print(df_train)
print(df_test)
df = pd.concat([df_train, df_test], axis=0)
df.to_csv(outDir + 'history.csv', index=False)
model.save(outDir + 'model')