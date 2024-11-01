import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import init_model #importing function from elsewhere in project

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images / 255.0
train_images = train_images / 255.0

# Create and compile model

# Define the ImageDataGenerator with augmentation options
datagen = ImageDataGenerator(
    rotation_range=20,        # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,    # Randomly translate images horizontally (fraction of total width)
    height_shift_range=0.1,   # Randomly translate images vertically (fraction of total height)
    shear_range=0.1,          # Shear intensity (shear angle in counter-clockwise direction in degrees)
    zoom_range=0.1,           # Randomly zoom into images
    horizontal_flip=True,     # Randomly flip images
    fill_mode='nearest'       # Fill pixels that are created after rotation or translation
)

# Fit the generator to your training data
datagen.fit(train_images)

model = init_model()

model.compile(metrics=['accuracy'],
              loss='sparse_categorical_crossentropy',
              optimizer='adam')


# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=3,                # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True   # Restore the model weights from the epoch with the best value of the monitored quantity
)

# Training model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=100, 
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping])

model.save("cirfar10_model2.keras") # using keras file format, more up to date




# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()