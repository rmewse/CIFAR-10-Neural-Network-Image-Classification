import tensorflow as tf
from tensorflow.keras import datasets

# This project uses the CIFAR-10 dataset, load the dataset
(train_images, train_labels), (test_images, test_labels) = dataset.cirfar10.load_data()

# Normalise all pixel values to be in range 0-1
test_images = test_images.astype("float32") / 255.0
train_images = test_images.astype("float32") / 255.0