# YOLO (v1) implementation from scratch on a subset of VOC dataset


#### data_loader.py: <br/>
Class to load images as well as their labels

#### loss.py: <br/>
Class defining the YOLO loss function

#### mean_average_precision.py: <br/>
Contains code to compute Mean Average Precision

#### model.py: <br/>
Class that implements YOLO model by extending tf.keras.models.Model

#### yolo_utils.py: <br/>
Contains code for various utility functions like Non-max suppression, transforming bounding boxes, plotting images and bounding boxes, etc

#### main.py: <br/>
Contains code to train and evaluate the YOLO model

#### voc_dataset: (not uploaded; create locally) <br/>
Folder containing the Pascal VOC dataset obtained from https://www.kaggle.com/aladdinpersson/pascalvoc-yolo

#### archive-code-to-be-deleted: <br/>
Ignore this folder; it contains old code
