## Folders:
ALEXNET, VGG, Inception_GoogleNet, RESNET, MobileNet, and LENET contain the corresponding research papers and their summaries.

datasets: (not uploaded; created locally)
A folder containing the ImageNette dataset. All models will read data from here.

test-images:
A folder containing few images to visualize outputs


## Files:
1) custom_model.py: 
   Contains generic code that applies transfer learning using a pre-trained model of your choice on your dataset.
2) vgg16.py, inceptionv3.py, resnet50.py, mobilenet.py:
   Depending on the model name you pass to custom_model.py, one of the model-specific files will be used.
3) data_utils.py:
   AlexNet takes multiple image patches of shape (227,227) from each training input image. This file generates such such patches from input images.
4) 2. ALEXNET/alexnet.py:
   AlexNet implementation from scratch using ImageNette dataset
5) 3. VGG/vgg16.py:
   VGG16 implementation from scratch using ImageNette dataset


## Dataset: 
Since "ImageNet" dataset is not publically available, all models use "Imagenette" dataset, which is taken from https://github.com/fastai/imagenette


## Transfer Learning:

Models trained using transfer-learning: (All the pre-trained models are used from tensorflow)
1) VGG16
2) Inception V3
3) RESNET 50
4) mobilenet_1.00_224

During transfer learning, the pre-trained model is first loaded without the top layers (the last pooling layer, FC layers, and the output layer). This model is then frozen. Later, Pooling, FC, and output layers are added as per requirement, and this model is trained for few epochs.

Once done, last few layers of the pre-trained model are unfrozen and then the model is trained again with a smaller learning rate. This helps adapt the pre-trained model for our image classification problem.
