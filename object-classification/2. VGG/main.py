from  data_loader import DataLoader
from vgg16 import VGG16
import tensorflow.keras as K
import sys


assert len(sys.argv) > 2, f"Pass learning rate and epochs at least; \
							in addition you can pass no. of units in dense layers and dropout rate"

lr = float(sys.argv[1])	# 0.005
epochs = int(sys.argv[2])			# 30
input_shape = (224, 224, 3)

if len(sys.argv) > 3:
	dense_units = int(sys.argv[2])
	drop = float(sys.argv[3])
else
	dense_units = 1024
	drop = 0.5

print("*"*10, f"LR-{lr}-epochs-{epochs}-units-{dense_units}-drop-{drop}", "*"*10)

#load training and validation data
ds_loader = DataLoader("..\\datasets\\imagenette2-320\\train",
							"..\\datasets\\imagenette2-320\\train",
							"..\\datasets\\imagenette2-320\\val")
train_ds, val_ds = ds_loader.load_train_val_data(input_shape[:2])

# create model
vgg_model = VGG16()
vgg_model.create()
vgg_model.plot_model('mymodel.png')

optimizer = K.optimizers.SGD(learning_rate=lr, momentum=0.9)
# loss = 
vgg_model.configure(optimizer)
vgg_model.train(train_ds, val_ds, epochs)
vgg_model.save_model(f"{lr}-{epochs}")

test_ds = ds_loader.load_test_data(input_shape[:2])
print(vgg_model.test(test_ds))