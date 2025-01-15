import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet50
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_roc_curve(y_true, y_pred, title="ROC Curve"):
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def save_results_to_csv(results, filepath="results.csv"):
    """Saves results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def grad_cam(model, img_array, layer_name="conv5_block3_out"):
    """Generates a Grad-CAM heatmap for ResNet50."""
    
    # Convert img_array to a TensorFlow tensor if it's not already
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Access the ResNet50 part of the model
    resnet50_layer = model.get_layer("resnet50")  # This is the base ResNet50 model
    
    # Ensure that you get the output of the last convolutional block
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,  # Use model.input
        outputs=[resnet50_layer.get_layer('resnet50').output, model.outputs]  # Use ResNet50 conv layer output
    )
    
    # Use GradientTape to compute the gradient
    with tf.GradientTape() as tape:
        tape.watch(img_array)  # Watch the image array
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])  # Find the class index with highest prediction
        loss = predictions[:, class_idx]  # Compute the loss for that class

    grads = tape.gradient(loss, conv_output)  # Get the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Pool the gradients across the spatial dimensions
    conv_output = conv_output[0]  # Select the first image's output
    heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)  # Weight the channels
    heatmap = tf.maximum(heatmap, 0)  # Remove negative values
    heatmap = heatmap / tf.math.reduce_max(heatmap)  # Normalize the heatmap

    return heatmap.numpy()  # Return the heatmap as a numpy array



def overlay_heatmap(original_img, heatmap, output_path):
    """Overlays the heatmap on the original image and saves the result."""
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap_img = tf.keras.preprocessing.image.array_to_img(heatmap)
    heatmap_img = heatmap_img.resize(original_img.size)

    # Convert heatmap to RGB and overlay
    heatmap_colored = np.array(heatmap_img.convert("RGB"))
    overlay = 0.4 * heatmap_colored + 0.6 * np.array(original_img)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Save overlay
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, overlay)
    print(f"Heatmap saved to {output_path}")


def analyze_test_images(test_dir, model):
    print(f"Analyzing test images in: {test_dir}...")

    # Initialize model input shape with a dummy call
    dummy_input = np.zeros((1, 224, 224, 3))  # Adjust shape if necessary
    _ = model.predict(dummy_input)

    categories = ['tb_positive', 'tb_negative']
    results = {"image_path": [], "true_label": [], "predicted_label": [], "confidence": []}
    y_true = []
    y_pred = []
    y_confidences = []

    threshold = 0.5

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

            # Predict and determine label
            prediction = model.predict(img_array)[0][0]
            predicted_label = "tb_positive" if prediction > threshold else "tb_negative"

            results["image_path"].append(image_path)
            results["true_label"].append(category)
            results["predicted_label"].append(predicted_label)
            results["confidence"].append(prediction)

            y_true.append(1 if category == "tb_positive" else 0)
            y_pred.append(1 if predicted_label == "tb_positive" else 0)
            y_confidences.append(prediction)

            print(f"Image: {image_file}, Predicted: {predicted_label}, Confidence: {prediction:.2f}")

            # Generate and save Grad-CAM heatmap
            heatmap = grad_cam(model, img_array)
            overlay_heatmap(img, heatmap, output_path=f"heatmaps/{category}_{image_file}")

    # Calculate and display metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=categories))

    plot_roc_curve(y_true, y_confidences, title="TB Detection ROC Curve")
    plot_confusion_matrix(y_true, y_pred, classes=categories)

    save_results_to_csv(results)



def main():
    # Print all layers to find the correct layer name for Grad-CAM
    
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