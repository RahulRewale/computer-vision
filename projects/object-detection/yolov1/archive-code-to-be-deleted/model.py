import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as tfl


def create_model(input_shape, S=7, B=2, classes=20):
	inp = K.Input(shape=input_shape)
	
	out = tfl.Conv2D(filters=64, kernel_size=(7, 7), padding='same', 
					strides=2, activation=tfl.LeakyReLU(alpha=0.1)) (inp)
	out = tfl.MaxPool2D(pool_size=(2, 2), strides=2) (out)

	out = tfl.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.MaxPool2D(pool_size=(2, 2), strides=2) (out)

	out = tfl.Conv2D(filters=128, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=256, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.MaxPool2D(pool_size=(2, 2), strides=2) (out)

	out = tfl.Conv2D(filters=256, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=256, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=256, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=256, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.MaxPool2D(pool_size=(2, 2), strides=2) (out)

	out = tfl.Conv2D(filters=512, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=512, kernel_size=(1, 1),
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', 
					strides=2, activation=tfl.LeakyReLU(alpha=0.1)) (out)

	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)
	out = tfl.Conv2D(filters=1024, kernel_size=(3, 3), padding='same',
					activation=tfl.LeakyReLU(alpha=0.1)) (out)

	out = tfl.Flatten() (out)
	out = tfl.Dense(units=1024, activation=tfl.LeakyReLU(alpha=0.1)) (out)
	# out = tfl.Dropout(0.3) (out)
	out = tfl.Dense(units=S*S*(classes+B*5)) (out)
	y = tfl.Reshape((S, S, classes+B*5)) (out)

	model = K.Model(inp, y)
	K.utils.plot_model(model, "model.png", show_shapes=True)

	return model


if __name__ == "__main__":
	model = create_model((448, 448, 3))

