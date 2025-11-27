import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir='data', batch_size=32, img_size=(224, 224)):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'Training'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'Testing'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())
    return train_generator, test_generator, class_names