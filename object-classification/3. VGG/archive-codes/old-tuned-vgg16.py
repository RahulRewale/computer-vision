import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt

vggModel = K.applications.VGG16(include_top=False, weights='imagenet', input_shape=(227, 227, 3))
vggModel.trainable = False
# print(vggModel.summary())


inp = K.Input(shape=(227, 227, 3))
x = K.applications.vgg16.preprocess_input(inp)
x = vggModel(x)
x = K.layers.Flatten()(x)
x = K.layers.Dense(units=1024, activation='relu')(x)
x = K.layers.Dense(units=1024, activation='relu')(x)
out = K.layers.Dense(units=10, activation='softmax')(x)
model = K.Model(inputs=inp, outputs=out)


trainDS = K.preprocessing.image_dataset_from_directory(
			directory = '..\\2. ALEXNET\\datasets\\augmented',
			label_mode = 'categorical', 
			batch_size = 128, 
			image_size = (227, 227)
			)
# normalize pixels
trainDS = trainDS.map(lambda x, y: (x/255.0, y))


valTestDS = K.preprocessing.image_dataset_from_directory(
			directory = '..\\2. ALEXNET\\datasets\\imagenette2-320\\val',
			label_mode = 'categorical',
			batch_size = 128,
			image_size = (227, 227)
			)

noOfExamples = tf.data.experimental.cardinality(valTestDS).numpy()
valDS = valTestDS.take(noOfExamples/2)		# first 50%
testDS = valTestDS.skip(noOfExamples/2)		# remaining 50%

# normalize pixels
valDS = valDS.map(lambda x, y: (x/255.0, y))
testDS = testDS.map(lambda x, y: (x/255.0, y))


img = K.preprocessing.image.load_img('dog.jpeg', target_size=(227, 227))
#C:\\Users\\Rahul\\Desktop\\PersonalDevelopment\\GitHub\\learning-material\\cv\\Papers\\3. VGG
arr = K.preprocessing.image.img_to_array(img)
arr = arr[np.newaxis, ...]
# newarr = K.applications.vgg16.preprocess_input(arr)

pred = model.predict(arr)
print("*"*10, pred, "*"*10)
# print(model.evaluate(testDS))


print('\n\n\n')
# decode_prediction doesn't work here as we have classes that are different than the ones found
# in imagenet. Also, the no. of classes is also different
# label = K.applications.vgg16.decode_predictions(pred, top=3)
# you need to implement decode logic for your problem yourself
classes = ['Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
			'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute']

plt.subplot(2,2,1)
plt.imshow(arr[0]/255.0)
plt.axis('off')
plt.title(classes[np.argmax(pred[0])])
plt.show()


# Now train the newly added layers with a small learning rate

model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.SGD(learning_rate=0.3),
				metrics=['accuracy'])
history = model.fit(trainDS, epochs=5, validation_data=valDS)


pred = model.predict(arr)
print("*"*10, pred, "*"*10)
print(model.evaluate(testDS))
print('\n\n\n')

plt.subplot(2,2,2)
plt.imshow(arr[0]/255.0)
plt.axis('off')
plt.title(classes[np.argmax(pred[0])])
plt.show()



for layer in vggModel.layers[-4:]:
	layer.trainable = True

print(model.trainable)

model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.SGD(learning_rate=0.006),
				metrics=['accuracy'])
model.fit(trainDS, epochs=10, initial_epoch=history.epoch[-1], validation_data=valDS)

pred = model.predict(arr)
print("*"*10, pred, "*"*10)
print(model.evaluate(testDS))
plt.subplot(2,2,3)
plt.imshow(arr[0]/255.0)
plt.axis('off')
plt.title(classes[np.argmax(pred[0])])
plt.show()
