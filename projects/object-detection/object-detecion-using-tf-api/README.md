# Object Detection Using Tensorflow Object Detection API

This directory contains an Object Detection project that I implemented using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). I have used a Faster RCNN model from [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and fine-tuned it on my dataset.


### [Google Colab Notebook](https://colab.research.google.com/drive/1SC4b0aKE5CeOuBuc4vZibLlKizcc5ApQ?usp=sharing)
Google Colab link that contains the code as well as the outputs from the trained model.


### [Dataset](https://drive.google.com/file/d/1gj4G4kM3BYO-YZPi1D1536zmNaNHUa9D/view?usp=sharing)
Link to a zip file containing the training as well as testing dataset. All the images are downloaded from the Internet. 
There are 10 different classes of objects. The classes are as below:
1) Person
2) Lion
3) Cat
4) Guitar
5) Glasses
6) Sword
7) Shoe
8) Reindeer
9) Pen
10) Cricket Bat <br/>

For labeling images, I used [LabelImg](https://github.com/tzutalin/labelImg)
<br/>

### Training
Trained the model for 10k training steps. Anchor boxes were selected manually by trying various values. Since the objects are of greatly varying shapes, I ended up using large number of anchor boxes. A better way to select anchor boxes is to use K-means algorithm, which I am planning to work on. <br/>
The anchor boxes, learning rate, confidence score and IOU thresholds, and other config details can be found in pipeline.config file.

### Results
Achieved an average precision of 0.551 @IoU=0.50:0.95 and 0.810 @IoU=0.50 <br/>
Achieved an average recall of 0.609 @IoU=0.50:0.95

### Resources
While creating this project, I have referred below resources: <br/>
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html <br/>
https://www.youtube.com/watch?v=yqkISICHH-U
