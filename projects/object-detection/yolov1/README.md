# YOLO (v1) Implementation from scratch on a subset of VOC dataset


data_loader.py:
A class containing code to load images as well as their labels.

loss.py:
A class defining the YOLO loss function.

mean_average_precision.py:
Contains code to compute Mean Average Precision

model.py:
A class that implements YOLO model by extending tf.keras.models.Model

yolo_utils.py:
Contains code for various utility functions like Non-max suppression, transforming bounding boxes, plotting images and bounding boxes, etc.

main.py:
Contains code to train and evaluate the YOLO model

voc_dataset: (not uploaded; create locally)
A folder containing the Pascal VOC dataset obtained from https://www.kaggle.com/aladdinpersson/pascalvoc-yolo


archive-code-to-be-deleted:
Ignore this folder; it contains old code.