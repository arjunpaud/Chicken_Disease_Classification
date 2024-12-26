import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

model = load_trained_model("model/model.h5")

# Define the class labels (adjust according to your model)
class_labels = ['Normal', 'Cocidocisis']  # Example labels, update as needed

# Define a function to preprocess the uploaded image
def preprocess_image(img, target_size):
    # Ensure the image has 3 channels (RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Streamlit app title
st.title("Image Classification with VGG16-based CNN")

# Upload an image
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)  # Adjusted image display
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img, target_size=(224, 224))
    
    # Make predictions
    predictions = model.predict(preprocessed_img)
    st.write("Raw Predictions:", predictions)
    
    # Assuming the model outputs class probabilities, display the most likely class
    class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[class_index]
    st.write(f"Predicted Class: {predicted_class} ({class_index})")
    
    # Display class-level probabilities
    probabilities = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    st.write("Class-level probabilities:")
    st.json(probabilities)
