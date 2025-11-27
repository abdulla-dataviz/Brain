# load_member2_model.py
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda
import tensorflow as tf

def load_member2_model():
    print("Loading Member 2's 92%+ accuracy model...")
    original_model = load_model("models/best_model.keras")
    print("Model loaded successfully!")
    
    inputs = Input(shape=(224, 224, 3))
    x = Lambda(lambda img: tf.image.resize(img, (128, 128)))(inputs)
    outputs = original_model(x)
    compatible_model = Model(inputs=inputs, outputs=outputs)
    
    print("Model now accepts 224x224 images -> auto-resized to 128x128")
    return compatible_model