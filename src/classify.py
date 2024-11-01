import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load your trained model
model = keras.models.load_model('your_model.h5')  # Replace with your model filename

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Resize image to match input shape
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Load and preprocess the image
img_path = 'path_to_your_image.jpg'  # Replace with the path to your image
processed_image = load_and_preprocess_image(img_path)

# Predict the class
predictions = model.predict(processed_image)
predicted_class = np.argmax(predictions, axis=-1)
print(f'Predicted class: {predicted_class[0]}')

# Visualize the image
plt.imshow(image.load_img(img_path))
plt.title(f'Predicted class: {predicted_class[0]}')
plt.axis('off')
plt.show()
