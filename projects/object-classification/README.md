### *UPDATING THIS FOLDER* ###
# Object Classification Using AlexNet

This directory contains AlexNet that I implemented from scratch for classifying 10 different classes of objects. I made few modifications to the original AlexNet model to prevent overfitting on the dataset that I am using.

### [Google Colab Notebook](): 
Google Colab link that contains the code as well as the outputs from the trained model.

### [Imagenette Dataset](https://github.com/fastai/imagenette): 
Since Imagenet dataset is not publically available, I used Imagenette dataset, which is a subset of Imagenet. The dataset contains 10 classes of objects. The classes are as below:
1) Tench
2) English Springer
3) Cassette Player
4) Chain Saw
5) Church
6) French Horn
7) Garbage Truck
8) Gas Pump
9) Golf Ball
10) Parachute


### Model:
The model has 64 neurons in fully connected layers (as opposed to 4096 in original Alexnet) and dropout of 0.5. All layers, except the output layer, use ReLU activation. The output layer has 10 neurons with Softmax activation.
Trained the model for 25 epochs using Adam optimizer with initial learning rate of 0.0001. The learning rate was reduced over time as and when validation loss stagnated. The model uses weight decay of 0.00001.


### Training:
Training dataset contains 9469 images. For training, 16 crops of (227, 227) were extracted from each training image. These 16 crops along with their mirror reflections were used for training.


### Validation and Testing:
For validation and testing, 5 crops were taken from each image. 4 crops from four corners and the 5th one from the center of the image. These 5 crops and their mirror reflections were used for validation and testing. Accuracy was calculated by taking average of the outputs of these 10 crops per image.


### Results (epoch 20 of 25): <br/>
Training Metrics: Loss 0.26260703802108765; Accuracy 0.8897388577461243 <br/>
Validation Metrics: Loss 0.5777757167816162; Accuracy 0.8654513955116272 <br/>
Testing Metrics: Loss 0.6200770139694214; Accuracy 0.8489000797271729 <br/>
