import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress HDF5 warnings for cleaner output
warnings.filterwarnings("ignore", message=".*HDF5 file.*", category=UserWarning)

# GPU memory growth configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except Exception as e:
        print(f"Error setting GPU memory growth: {e}")

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/testing'
heatmap_dir = 'tb_heatmaps'
model_path = 'tb_detection_best_model.h5'

os.makedirs(heatmap_dir, exist_ok=True)

def prepare_dataset():
    print("Preparing dataset...")
    for directory in [train_dir, val_dir]:
        if not os.path.exists(directory):
            print(f"Dataset directory not found: {directory}")
            return False
    return True

def get_class_weights(generator):
    class_counts = generator.classes
    class_labels = np.unique(class_counts)
    class_weights = compute_class_weight('balanced', classes=class_labels, y=class_counts)
    return {i: weight for i, weight in enumerate(class_weights)}

def build_model(base_trainable=False):
    print("Building the model using ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = base_trainable
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    model.summary()
    return model

class ValidationStopper(Callback):
    def __init__(self, validation_data, target_accuracy=0.95):
        super().__init__()
        self.validation_data = validation_data
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_generator, val_labels = self.validation_data
        val_predictions = (self.model.predict(val_generator) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(val_labels, val_predictions)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        if accuracy >= self.target_accuracy:
            print(f"Target accuracy {self.target_accuracy:.4f} reached. Stopping training.")
            self.model.stop_training = True

def get_validation_data(generator):
    data, labels = [], []
    for _ in range(len(generator)):
        batch_data, batch_labels = next(generator)
        data.append(batch_data)
        labels.append(batch_labels)
    return np.vstack(data), np.hstack(labels)

def train_model():
    print("Training the model...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet50.preprocess_input,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        zoom_range=0.2, horizontal_flip=True, shear_range=0.15
    )
    val_datagen = ImageDataGenerator(preprocessing_function=resnet50.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
    )

    class_weights = get_class_weights(train_generator)
    
    val_data, val_labels = get_validation_data(val_generator)

    # Phase 1: Train top layers
    print("Phase 1: Training top layers...")
    model = build_model(base_trainable=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    validation_stopper = ValidationStopper(validation_data=(val_data, val_labels), target_accuracy=0.95)
    checkpoint_phase1 = ModelCheckpoint('tb_detection_best_model_phase1.h5', monitor='val_loss', save_best_only=True)
    model.fit(
        train_generator, epochs=15, validation_data=val_generator,
        class_weight=class_weights, callbacks=[early_stopping, validation_stopper, checkpoint_phase1]
    )

    # Phase 2: Fine-tuning
    print("Phase 2: Fine-tuning the base model...")
    model = load_model('tb_detection_best_model_phase1.h5')
    for layer in model.layers[0].layers[-50:]:
        layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    validation_stopper = ValidationStopper(validation_data=(val_data, val_labels), target_accuracy=0.98)
    checkpoint_final = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    model.fit(
        train_generator, epochs=5, validation_data=val_generator,
        class_weight=class_weights, callbacks=[early_stopping, validation_stopper, checkpoint_final]
    )

    print(f"Model saved as '{model_path}'")
    return model

def analyze_test_images(image_dir, model):
    print(f"Analyzing test images in: {image_dir}...")
    threshold = 0.0002
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224), color_mode='rgb')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "TB Detected" if prediction > threshold else "No TB Detected"
        confidence = prediction * 100 if prediction > threshold else (1 - prediction) * 100

        print(f"Image: {image_file}, {label}, {confidence:.2f}% confidence")

        with open("test_results.txt", "a") as f:
            f.write(f"Image: {image_file}, {label}, {confidence:.2f}% confidence\n")

def main():
    print("Welcome to the Tuberculosis Detection Program!")
    if not prepare_dataset():
        return

    if os.path.exists(model_path):
        print("Loading existing model...")
        try:
            model = load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Training a new model...")
        model = train_model()

    analyze_test_images(test_dir, model)

if __name__ == "__main__":
    main()
