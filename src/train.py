import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from model import init_model #importing function from elsewhere in project

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images / 255.0
train_images = train_images / 255.0

# Create and compile model

model = init_model()

model.compile(metrics=['accuracy'],
              loss='sparse_categorical_crossentropy',
              optimizer='adam')

# Training model

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

model.save("cirfar10_model.keras") # using keras file format, more up to date




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