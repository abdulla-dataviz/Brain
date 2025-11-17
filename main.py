import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import shutil
import cv2  # For image preprocessing

# Suppress oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Attempt to import imblearn for oversampling; use fallback if not available
try:
    from imblearn.over_sampling import RandomOverSampler
    oversampling_available = True
except ImportError:
    print("Warning: imbalanced-learn not found. Oversampling will be skipped.")
    oversampling_available = False

# ---------------------------
# Build CNN (Enhanced Architecture)
# ---------------------------
def build_cnn(input_shape=(128, 128, 3), num_classes=4):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))  # New layer
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))  # Increased units
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return model

# ---------------------------
# Data Paths
# ---------------------------
dataset_base_path = r"D:\Brain Tumour Detection\CNN Model\Dataset"
train_base_path = os.path.join(dataset_base_path, "Training")
test_path = os.path.join(dataset_base_path, "Testing")

# Debug: Check if paths exist
print(f"Checking dataset base path: {dataset_base_path}")
if not os.path.exists(dataset_base_path):
    print(f"Error: Dataset base path does not exist: {dataset_base_path}")
    exit(1)
print(f"Checking training path: {train_base_path}")
if not os.path.exists(train_base_path):
    print(f"Error: Training path does not exist: {train_base_path}")
    exit(1)
print(f"Checking testing path: {test_path}")
if not os.path.exists(test_path):
    print(f"Error: Testing path does not exist: {test_path}")
    exit(1)

# ---------------------------
# Create Validation Split from Training Data with Optional Oversampling
# ---------------------------
def create_val_split(train_base_path, val_split=0.2):
    if not os.path.exists(train_base_path):
        print(f"Error: Directory not found: {train_base_path}")
        exit(1)
    all_images = []
    labels = []
    class_names = sorted([d for d in os.listdir(train_base_path) if os.path.isdir(os.path.join(train_base_path, d))])
    print(f"Found class folders: {class_names}")
    if not class_names:
        print("Error: No class folders found in Training directory.")
        exit(1)
    
    # Count and load images with their paths
    class_counts = {}
    image_data = []
    original_images = {}
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_base_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(image_files)
        original_images[class_name] = [os.path.join(class_path, f) for f in image_files]
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) / 255.0
            image_data.append(img.flatten())
            labels.append(class_idx)
            all_images.append(img_path)
    
    print("Number of images per class:", class_counts)
    print(f"Total images: {len(all_images)}")
    
    # Oversample minority classes if available
    if oversampling_available:
        ros = RandomOverSampler(random_state=42)
        image_data_2d = np.array(image_data)
        labels_2d = np.array(labels).reshape(-1, 1)
        image_data_res, labels_res = ros.fit_resample(image_data_2d, labels_2d)
        labels_res = labels_res.flatten()
        
        # Map oversampled labels to original image paths
        all_images_res = []
        for label in labels_res:
            class_name = class_names[label]
            # Select the original image path based on the label's position in the oversampled set
            original_idx = np.where(np.array(labels) == label)[0]
            if len(original_idx) > 0:
                base_idx = original_idx[0] % len(original_images[class_name])
                all_images_res.append(original_images[class_name][base_idx])
            else:
                # Fallback to first image if no match (should not occur with proper oversampling)
                all_images_res.append(original_images[class_name][0])
        print(f"Oversampled total images: {len(all_images_res)}")
    else:
        all_images_res = all_images
        labels_res = labels
    
    # Check if dataset is too small for splitting (minimum 4 images per class after oversampling)
    min_samples = min(len([l for l in labels_res if l == i]) for i in range(len(class_names)))
    if len(all_images_res) < len(class_names) * 4 or min_samples < 2:
        print(f"Warning: Dataset too small ({len(all_images_res)} images) or imbalanced after oversampling. Using test set for validation.")
        return train_base_path, test_path, class_names
    
    # Perform non-stratified split
    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images_res, labels_res, test_size=val_split, random_state=42
    )
    
    # Create validation directories
    val_base_path = os.path.join(dataset_base_path, "Validation")
    os.makedirs(val_base_path, exist_ok=True)
    for class_idx, class_name in enumerate(class_names):
        os.makedirs(os.path.join(val_base_path, class_name), exist_ok=True)
    
    # Copy val images with existence check
    for img_path, label in zip(val_images, val_labels):
        class_name = class_names[label]
        dest_path = os.path.join(val_base_path, class_name, os.path.basename(img_path))
        if os.path.exists(img_path) and not os.path.exists(dest_path):
            shutil.copy(img_path, dest_path)
        else:
            print(f"Warning: Source image {img_path} not found or already copied. Skipping.")
    
    print(f"Created validation split: {len(train_images)} train images, {len(val_images)} val images")
    return train_base_path, val_base_path, class_names

try:
    train_path, val_path, class_names = create_val_split(train_base_path)
except Exception as e:
    print(f"Error in create_val_split: {str(e)}")
    exit(1)

# ---------------------------
# Data Generators (Enhanced Augmentation)
# ---------------------------
try:
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3,
        shear_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_path, target_size=(128, 128),
                                                  batch_size=32, class_mode='categorical', shuffle=True)
    val_gen = val_datagen.flow_from_directory(val_path, target_size=(128, 128),
                                              batch_size=32, class_mode='categorical')
    
    if val_gen.samples == 0:
        print("Error: Validation dataset is empty. Check the split.")
        exit(1)

    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")
except Exception as e:
    print(f"Error setting up data generators: {str(e)}")
    exit(1)

# ---------------------------
# Training (With Callbacks and Class Weights)
# ---------------------------
model = build_cnn(num_classes=num_classes)

# Collect labels for class weights
train_labels = []
for _ in range(len(train_gen)):
    _, labels = next(train_gen)
    train_labels.extend(np.argmax(labels, axis=1))
train_labels = np.array(train_labels)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(enumerate(class_weights * 2))  # Double weight for minority classes

# Reset generator
train_gen = train_datagen.flow_from_directory(train_path, target_size=(128, 128),
                                              batch_size=32, class_mode='categorical', shuffle=True)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

try:
    history = model.fit(train_gen, validation_data=val_gen, epochs=50,
                        class_weight=class_weight_dict, callbacks=[early_stopping, lr_scheduler])
    model.save("brain_tumor_model.keras")
except Exception as e:
    print(f"Error during training: {str(e)}")
    exit(1)

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# ---------------------------
# Training Plots
# ---------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Model Loss")
plt.tight_layout()
plt.show()

# ---------------------------
# Evaluation
# ---------------------------
try:
    test_gen = val_datagen.flow_from_directory(test_path, target_size=(128, 128),
                                               batch_size=32, class_mode='categorical', shuffle=False)
    
    if test_gen.samples == 0:
        print("Error: Test dataset is empty. Please check 'Testing' directory.")
        exit(1)

    loss, acc, recall, precision = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test Recall: {recall*100:.2f}%")
    print(f"Test Precision: {precision*100:.2f}%")

    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
    exit(1)

# ---------------------------
# Prediction Function (Enhanced)
# ---------------------------
def predict_image(img_path, model, class_indices=None):
    if class_indices is None:
        class_indices = train_gen.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    try:
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        return f"Predicted: {class_labels[predicted_class]} (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Example use
print(predict_image(r"D:\Brain Tumour Detection\CNN Model\Dataset\Testing\glioma\image(1).jpg", model))