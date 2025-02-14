import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet50
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import warnings

warnings.filterwarnings("ignore", message=".*HDF5 file.*", category=UserWarning)

# Path to your saved model
model_path = 'tb_model.h5'

# Set threshold for classifying TB positive vs negative
THRESHOLD = 0.5

# Load the model once at the start
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses the image.
    """
    try:
        # Load the image using PIL
        img = Image.open(image_path).convert('RGB')
        # Keep a copy for display purposes
        display_image = img.copy()
        # Resize for model input
        img = img.resize(target_size)
        # Convert to numpy array
        img_array = np.array(img)
        # Preprocess using ResNet50's preprocess_input
        img_array = resnet50.preprocess_input(img_array)
        # Expand dims to create batch of size 1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, display_image
    except Exception as e:
        messagebox.showerror("Error", f"Error processing image: {e}")
        return None, None


def predict_tb(image_path):
    """
    Loads an image, preprocesses it, predicts using the model,
    and returns the prediction probability.
    """
    img_array, display_image = preprocess_image(image_path)
    if img_array is None:
        return None, None
    # Predict probability of TB positive (assuming model outputs a single probability)
    prediction = model.predict(img_array)[0][0]
    return prediction, display_image


class TBDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TB Detection from Lung X-ray")
        self.geometry("600x700")
        self.configure(bg="#f0f0f0")

        # Title Label
        title_label = tk.Label(self, text="Tuberculosis Detection", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)

        # Button to select an image
        select_button = tk.Button(self, text="Select Lung X-ray", font=("Helvetica", 14), command=self.select_image)
        select_button.pack(pady=10)

        # Label to display selected image
        self.image_label = tk.Label(self, bg="#f0f0f0")
        self.image_label.pack(pady=10)

        # Label to display prediction result
        self.result_label = tk.Label(self, text="", font=("Helvetica", 14), bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def select_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(
            title="Select an X-ray image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        # Predict TB probability
        prediction, display_image = predict_tb(image_path)
        if prediction is None:
            return

        # Compute probabilities for TB positive and negative
        prob_positive = prediction
        prob_negative = 1 - prediction

        # Determine predicted label
        predicted_label = "TB Positive" if prediction > THRESHOLD else "TB Negative"

        # Update result label with probabilities
        result_text = (
            f"Prediction: {predicted_label}\n"
            f"TB Positive: {prob_positive*100:.2f}%\n"
            f"TB Negative: {prob_negative*100:.2f}%"
        )
        self.result_label.config(text=result_text)

        # Update image display (resize to fit in the GUI)
        display_image = display_image.resize((400, 400))
        tk_image = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image  # keep a reference


if __name__ == "__main__":
    app = TBDetectionApp()
    app.mainloop()
