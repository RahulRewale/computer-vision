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

	return trainDS.take(50), valDS.take(30), testDS


# 5e-5, 8e-5, 6e-5 (96% test accuracy) by tuning last four layers of VGG16

# By tuning last 8 layers of VGG16
# 1e-4 (0.9762/0.9677/0.9449)
# 8e-5 ()

loadLR  = float(sys.argv[1])	# learning rate for loading the tuned model
trainLR = float(sys.argv[2])	# learning rate for fine-tuning

# layer from which to train, counted backward;
# try 4 or 8
layerNo = int(sys.argv[3])

outDir = f'.\\models\\tunedModel-{loadLR}-{trainLR}-{layerNo}\\'
os.makedirs(outDir)

classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']


model = K.models.load_model(f'.\\models\\trainedmodel-{loadLR}\\model')
# print(model.summary())
resnetModel = model.get_layer('resnet50')
# print(resnetModel.summary())

resnetModel.trainable=True

# -32: conv5_block1_1_conv
# -20: conv5_block2_1_conv
for layer in resnetModel.layers[:-layerNo]:
	layer.trainable = False


trainDS, valDS, testDS = loadData()

model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.SGD(learning_rate=trainLR),
 				metrics=['accuracy'])


img = K.preprocessing.image.load_img('test-images\\cassette.jpeg', target_size=(227, 227))
arr = K.preprocessing.image.img_to_array(img)
arr = arr[np.newaxis, ...]


print("\n\nFine tuning the model")
earlyStop = K.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(trainDS, epochs=15, validation_data=valDS, callbacks=[earlyStop])

print("\n\nTesting the model")
pred = model.predict(arr)
print("*"*10, pred, "*"*10)
plt.imshow(arr[0]/255.0)
plt.axis('off')
plt.title(classes[np.argmax(pred[0])])
plt.show()

testResults = model.evaluate(testDS)
print(testResults)


df_train = pd.DataFrame(history.history)
df_test = pd.DataFrame([testResults], columns=['test_loss', 'test_acc'])
print(df_train)
print(df_test)
df = pd.concat([df_train, df_test], axis=0)
df.to_csv(outDir + 'history.csv', index=False)

model.save(outDir + 'model')