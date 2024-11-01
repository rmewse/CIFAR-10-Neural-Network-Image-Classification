import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# This project uses the CIFAR-10 dataset, load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalise all pixel values to be in range 0-1
test_images = test_images.astype("float32") / 255.0
train_images = test_images.astype("float32") / 255.0

def init_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(20, activation='softmax'))
    return model