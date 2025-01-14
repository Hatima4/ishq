import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet50
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import warnings

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
test_dir = 'dataset/testing'
model_path = 'tb_model.h5'

def analyze_test_images(test_dir, model):
    print(f"Analyzing test images in: {test_dir}...")

    categories = ['tb_positive', 'tb_negative']
    results = {category: {"correct": 0, "total": 0} for category in categories}
    threshold = 0.05

    for category in categories:
        folder_path = os.path.join(test_dir, category)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        print(f"Processing folder: {category}")
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                img = tf.keras.preprocessing.image.load_img(
                    image_path, target_size=(224, 224), color_mode='rgb'
                )
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = resnet50.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            predicted_label = "tb_positive" if prediction > threshold else "tb_negative"

            # Update results
            if predicted_label == category:
                results[category]["correct"] += 1
            results[category]["total"] += 1

            print(f"Image: {image_file}, Predicted: {predicted_label}, Confidence: {prediction:.2f}")

    # Calculate and display accuracy
    total_correct = sum(results[cat]["correct"] for cat in categories)
    total_images = sum(results[cat]["total"] for cat in categories)

    print("\nResults:")
    for category in categories:
        correct = results[category]["correct"]
        total = results[category]["total"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"{category}: {correct}/{total} ({accuracy:.2f}% accuracy)")

    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    print(f"\nOverall Accuracy: {total_correct}/{total_images} ({overall_accuracy:.2f}%)")


def main():
    print("Welcome to the Tuberculosis Detection Program!")

    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print("Loading model...")
    model = load_model(model_path)

    analyze_test_images(test_dir, model)


if __name__ == "__main__":
    main()
