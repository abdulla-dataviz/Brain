# test_upload.py  ← Run this file to upload and test any image
import os
from validation import validate_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from load_member2_model import load_member2_model

def predict_uploaded_image():
    print("=== BRAIN TUMOR DETECTION – UPLOAD YOUR IMAGE ===\n")
    
    # Ask for image path
    path = input("Enter full image path (or drag & drop into this window):\n").strip()
    path = path.strip('"\'')  # remove quotes if dragged
    
    if not os.path.exists(path):
        print("File not found! Please check the path.")
        return
    
    print(f"\nChecking: {os.path.basename(path)}")
    print("-" * 50)
    
    # Step 1: Member 3's validation
    is_valid, message = validate_image(path)
    if not is_valid:
        print(f"REJECTED: {message}")
        return
    
    print("VALID BRAIN MRI! Running 92% accurate model...")
    
    # Step 2: Load Member 2's model
    model = load_member2_model()
    
    # Step 3: Predict
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    result = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    print("\n" + "="*50)
    print(f"     RESULT: {result.upper()}")
    print(f" CONFIDENCE: {confidence:.2f}%")
    print("="*50)

# Run it!
if __name__ == "__main__":
    predict_uploaded_image()
    input("\nPress Enter to exit...")