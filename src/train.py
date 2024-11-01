import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from model import init_model #importing function from elsewhere in project

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images / 255.0
train_images = train_images / 255.0

# Create and compile model

model = init_model()

model.compile(metrics=['accuracy'],
              loss='sparse_categorical_crossentropy',
              optimizer='adam')


# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=5,                # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True   # Restore the model weights from the epoch with the best value of the monitored quantity
)

# Training model

history = model.fit(
    train_images, 
    train_labels, 
    epochs=30, 
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