from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model with a relative path
model = load_model("brain_tumor_model.keras")

# Specify the path to the image you want to predict
img_path = r"D:\Brain Tumour Detection\CNN Model\Dataset\Testing\meningioma\Te-me_0012.jpg"# Update this to an existing file
img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
class_indices = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}  # Matches main.py order
predicted_class = np.argmax(prediction, axis=1)[0]
confidence = np.max(prediction)

# Print the result
print(f"Predicted: {class_indices[predicted_class]} (Confidence: {confidence:.2f})")