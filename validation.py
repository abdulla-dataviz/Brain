import cv2
import numpy as np
import os
import torch

# Input Validation Module
def validate_image(file_path):
    """
    Validates the uploaded image based on format, size, grayscale content, and basic quality.
    
    Args:
        file_path (str): Path to the image file.
    
    Returns:
        tuple: (bool, str) - Validation status and message.
    """
    # File format checking
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        return False, "Invalid file format. Only JPG and PNG allowed."
    
    # Try to open the image
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image")
    except Exception as e:
        return False, f"Corrupted file or cannot open: {str(e)}"
    
    # Image size validation (minimum 100x100)
    h, w, c = img.shape
    if h < 100 or w < 100:
        return False, "Image too small."
    
    # Grayscale content detection (MRI vs regular photos)
    b, g, r = cv2.split(img)
    if not (np.allclose(b, g) and np.allclose(g, r)):
        return False, "Image is not grayscale. Please upload brain MRI images."
    
    # Basic quality checks
    if np.mean(img) < 10 or np.mean(img) > 245:
        return False, "Image quality too low (blank or overexposed)."
    
    if np.std(img) < 5:
        return False, "Image has no content (uniform)."
    
    return True, "Image is valid."

# Enhanced Prediction Pipeline
def predict_tumor(image_path, model_path):
    """
    Predicts if the image shows a brain tumor using the loaded model.
    Includes input validation, preprocessing, confidence scoring, and edge case handling.
    
    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the saved PyTorch model.
    
    Returns:
        dict: Prediction result with confidence or error message.
    """
    # Input validation
    is_valid, message = validate_image(image_path)
    if not is_valid:
        return {"error": message}
    
    # Load saved model (assuming PyTorch model)
    try:
        model = torch.load(model_path)
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    # Preprocess the image
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image for preprocessing")
        
        img = cv2.resize(img, (224, 224))  # Assuming model input size is 224x224
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img_tensor = torch.from_numpy(img).float()
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}
    
    # Inference
    try:
        with torch.no_grad():
            output = model(img_tensor)
        
        probs = torch.softmax(output, dim=1)
        max_prob = torch.max(probs).item()
        class_idx = torch.argmax(probs).item()
        
        classes = ["no_tumor", "tumor"]  # Assuming binary classification
        
        # Confidence threshold tuning (60% cutoff for non-brain rejection)
        if max_prob < 0.6:
            return {
                "result": "rejected",
                "confidence": round(max_prob * 100, 2),
                "message": "Image does not appear to be a valid brain MRI (low confidence)."
            }
        
        return {
            "result": classes[class_idx],
            "confidence": round(max_prob * 100, 2)
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

# Test Cases
# Note: These are example test cases. In a real scenario, use unittest or pytest.
# Here, we describe them and provide simulation code for demonstration.

"""
Test Case 1: Valid brain MRI (grayscale)
- Create a simulated grayscale image.
- Expect: Valid, proceeds to prediction.

Test Case 2: Regular photo (color, e.g., cat photo)
- Create a color image.
- Expect: Fails grayscale check.

Test Case 3: Blank image
- All white or black.
- Expect: Fails quality check.

Test Case 4: Uniform image (low variance)
- Constant color.
- Expect: Fails std check.

Test Case 5: Small image
- Size < 100x100.
- Expect: Fails size check.

Test Case 6: Corrupted file
- Non-image file with .jpg extension.
- Expect: Fails open.

Test Case 7: Wrong format (e.g., .txt)
- Expect: Fails format check.

For prediction tests: Since model is placeholder, test with dummy model.
"""

# Example simulation code for tests (can be run in your environment)
if __name__ == "__main__":
    # Create test files (simulation)
    # Grayscale MRI sim
    gray = np.random.uniform(50, 200, (200, 200)).astype(np.uint8)
    gray_rgb = cv2.merge([gray, gray, gray])
    cv2.imwrite('test_mri.jpg', gray_rgb)
    
    print("Test MRI:", validate_image('test_mri.jpg'))
    
    # Color sim
    color = np.random.randint(0, 255, (200, 200, 3)).astype(np.uint8)
    cv2.imwrite('test_cat.jpg', color)
    
    print("Test Cat:", validate_image('test_cat.jpg'))
    
    # Add more as needed...

    # For prediction, assume a model_path exists.
    # result = predict_tumor('test_mri.jpg', 'path/to/saved_model.pth')
    # print(result)