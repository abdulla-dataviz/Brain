from data_loader import load_data
from test_prediction import predict_and_visualize
from load_member2_model import load_member2_model   # ← NEW

if __name__ == "__main__":
    print("Loading data (your 224x224 pipeline)...")
    train_gen, test_gen, class_names = load_data()
    
    print("Loading Member 2's 92% accuracy model...")
    model = load_member2_model()   # ← This is the key line
    
    print("Generating prediction images (1 per class)...")
    predict_and_visualize(model, test_gen)
    
    print("DONE! Check the saved prediction images.")
    print("Your pipeline + Member 2's brain = 92% accuracy!")

    # =============================================
# MEMBER 3: FINAL SINGLE IMAGE PREDICTION WITH VALIDATION
# =============================================
from validation import validate_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_with_validation(image_path):
    # Step 1: Member 3 checks if it's a real MRI
    is_valid, message = validate_image(image_path)
    if not is_valid:
        print(f"REJECTED: {message}")
        return
    
    print("VALID MRI DETECTED! Running prediction with Member 2's 92% model...")
    
    # Step 2: Load Member 2's model (already fixed for 224x224)
    from load_member2_model import load_member2_model
    model = load_member2_model()
    
    # Step 3: Predict
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    result = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100
    
    print(f"PREDICTION: {result}")
    print(f"CONFIDENCE: {confidence:.2f}%")

# =============================================
# TEST MEMBER 3'S WORK (Run these!)
# =============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING MEMBER 3'S VALIDATION + MEMBER 2'S MODEL")
    print("="*60)
    
    predict_with_validation("test_mri.jpg")   # Should say VALID + prediction
    print("-" * 50)
    predict_with_validation("test_cat.jpg")   # Should say REJECTED (not grayscale)