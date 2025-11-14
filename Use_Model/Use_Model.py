import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("../Model/Brain_Tumor_Classifier_Model.keras")

# Preprocess function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))            # your model's input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)            # shape -> (1,128,128,3)
    return img

# Prediction function
def predict(image_path):
    img = preprocess_image(image_path)
    output = model.predict(img)

    print("Raw output:", output)

    # CASE 1: sigmoid output → example: [[0.82]]
    if output.shape[-1] == 1:
        prob = output[0][0]
        if prob > 0.5:
            print("Prediction: Tumor   | Confidence:", prob)
        else:
            print("Prediction: No Tumor | Confidence:", 1 - prob)

    # CASE 2: softmax output → example: [[0.1, 0.9]]
    else:
        classes = ["No Tumor", "Tumor"]
        index = np.argmax(output)
        print("Prediction:", classes[index], "| Confidence:", output[0][index])

# Test prediction
predict("C:/Users/BALAJI/OneDrive/Desktop/Roopak Raam/Projects/Brain Tumor Classification/brain_tumor_dataset/yes/Y20.jpg")
