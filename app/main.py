import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import json
from PIL import Image
import os

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_cnnmodel.h5')
model = tf.keras.models.load_model(model_path)
# Load the class names
# Load the class names
class_indices_path = os.path.join(os.path.dirname(__file__), 'class_indices.json')
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{class_indices_path}' was not found. Please check the path.")

# Reverse the class_indices to map indices back to their corresponding class labels
index_to_class = {v: k for k, v in class_indices.items()}

# Define a function to make predictions
def predict_disease(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize the image to match the model input size
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_category_name = index_to_class.get(predicted_class, "Unknown")
    
    # Return both the prediction and the prediction confidence (probability)
    confidence = np.max(predictions[0])
    return predicted_category_name, confidence

st.image("plantsheading.png", use_column_width=True)
# Page title and description with emojis on the same line
#st.title("🌿 Plant Disease Detection 🌿") 
# Description of the app
st.write("""
    **Welcome to the Plant Disease Detection App 🌿!**  
    Upload a picture of a plant leaf, and the app will analyze it using a deep learning model to detect potential diseases.  
    This app is designed to assist farmers and plant enthusiasts in identifying diseases early and taking appropriate action.
    """)

# Sidebar for additional information or input
st.sidebar.title("About the App")
st.sidebar.write("""
    This app uses a convolutional neural network (CNN) model to detect plant diseases from images.
    Upload an image and the model will predict the disease (if any).
""")
#st.sidebar.image("leafbackground.png", use_column_width=True)
st.sidebar.header("Instructions")
st.sidebar.write("""
    - Upload a clear image of a plant leaf.
    - Wait for the model to analyze and return predictions.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    prediction, confidence = predict_disease(uploaded_file)

    # Display the prediction
    st.write(f"### Prediction: **{prediction}**")
    st.write(f"### Confidence: **{confidence * 100:.2f}%**")

    # Display additional information about the disease
    disease_info = {
    # Apple
    "Apple___Apple_scab": "This leaf is diseased. Apply fungicides in early spring and improve air circulation around the tree.",
    "Apple___Black_rot": "This leaf is diseased. Remove infected branches, dispose of fallen debris, and apply fungicide as needed.",
    "Apple___Cedar_apple_rust": "This leaf is diseased. Plant rust-resistant apple varieties and apply fungicide during the early growing season.",
    "Apple___healthy": "This apple leaf appears healthy! Maintain regular pruning of the tree, apply balanced fertilizers, and monitor for early signs of disease or pests.",
    
    # Blueberry
    "Blueberry___healthy": "This blueberry leaf is in good health. Ensure the plant has acidic, well-drained soil, water regularly, and mulch to retain moisture and suppress weeds.",
    
    # Cherry (including sour)
    "Cherry_(including_sour)___Powdery_mildew": "This leaf is diseased. Apply sulfur-based fungicides and ensure good air circulation around the tree.",
    "Cherry_(including_sour)___healthy": "This cherry leaf is healthy. Provide good air circulation, prune regularly, and apply balanced fertilizers to maintain the tree’s vigor.",
    
    # Corn (maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "This leaf is diseased. Rotate crops and apply foliar fungicides when symptoms first appear.",
    "Corn_(maize)___Common_rust_": "This leaf is diseased. Apply fungicide and consider using rust-resistant corn varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "This leaf is diseased. Use resistant hybrids and apply fungicides at the first sign of disease.",
    "Corn_(maize)___healthy": "Corn leaves look healthy! Keep an eye on pests, provide adequate water, and use crop rotation to prevent soil-borne diseases.",
    
    # Grape
    "Grape___Black_rot": "This leaf is diseased. Prune affected vines and apply fungicides in early spring.",
    "Grape___Esca_(Black_Measles)": "This leaf is diseased. Limit pruning during rainy seasons and use appropriate fungicides.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "This leaf is diseased. Remove affected leaves and apply fungicide during wet seasons.",
    "Grape___healthy": "Grape leaves appear healthy. Prune the vines annually, ensure good airflow, and avoid overhead watering to reduce the risk of fungal diseases.",
    
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": "This leaf is diseased. Remove infected trees and control psyllids with insecticides to prevent spread.",
    "Orange___healthy": "This orange leaf is healthy. Provide consistent watering, monitor for pests, and apply citrus-specific fertilizer for optimal growth.",
    
    # Peach
    "Peach___Bacterial_spot": "This leaf is diseased. Use copper-based bactericides and select resistant peach varieties.",
    "Peach___healthy": "Peach leaf looks good! Prune to improve airflow, fertilize in early spring, and monitor for signs of fungal diseases or pests on nearby foliage.",
    
    # Pepper (bell)
    "Pepper,_bell___Bacterial_spot": "This leaf is diseased. Apply copper sprays and ensure seeds are disease-free.",
    "Pepper,_bell___healthy": "This bell pepper leaf is healthy. Ensure regular watering, apply balanced fertilizers, and monitor nearby leaves for any pests or diseases.",
    
    # Potato
    "Potato___Early_blight": "This leaf is diseased. Rotate crops and apply fungicides to manage spread.",
    "Potato___Late_blight": "This leaf is diseased. Use resistant varieties and apply fungicides regularly.",
    "Potato___healthy": "Potato leaves appear healthy. Hill soil around plants as they grow, rotate crops annually, and monitor for early signs of blight on the foliage.",
    
    # Raspberry
    "Raspberry___healthy": "Raspberry leaf is healthy. Prune in early spring, provide support as needed, and water consistently to promote healthy growth.",
    
    # Soybean
    "Soybean___healthy": "Soybean leaves are looking good! Practice crop rotation, monitor for pests, and avoid overcrowding to reduce disease risk.",
    
    # Squash
    "Squash___Powdery_mildew": "This leaf is diseased. Improve air circulation and apply fungicides to control the disease.",
    
    # Strawberry
    "Strawberry___Leaf_scorch": "This leaf is diseased. Remove infected leaves and apply fungicides as necessary.",
    "Strawberry___healthy": "Strawberry leaves are in good health. Remove old leaves, mulch to retain moisture, and monitor for signs of leaf scorch or pests.",
    
    # Tomato
    "Tomato___Bacterial_spot": "This leaf is diseased. Apply copper fungicide sprays and avoid overhead watering.",
    "Tomato___Early_blight": "This leaf is diseased. Rotate crops and apply fungicides at the first sign of infection.",
    "Tomato___Late_blight": "This leaf is diseased. Use resistant varieties and apply fungicides regularly.",
    "Tomato___Leaf_Mold": "This leaf is diseased. Increase ventilation and apply fungicide as necessary.",
    "Tomato___Septoria_leaf_spot": "This leaf is diseased. Prune affected leaves and apply fungicides to control spread.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "This leaf is diseased. Use insecticidal soap or miticides to control mites.",
    "Tomato___Target_Spot": "This leaf is diseased. Apply fungicides and remove affected foliage promptly.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "This leaf is diseased. Control whiteflies and remove infected plants to prevent spread.",
    "Tomato___Tomato_mosaic_virus": "This leaf is diseased. Disinfect tools and avoid handling plants if infected.",
    "Tomato___healthy": "This tomato leaf is healthy! Provide support, water consistently at the base of the plant, and inspect regularly to catch any early signs of disease."
}
   
    # Check if there's extra info for the predicted disease
    if prediction in disease_info:
        st.write(f"{disease_info[prediction]}")
    else:
        st.write("No additional information available for this disease.")
else:
    st.write("Please upload an image to get started.")

# Footer with some styling
st.markdown("""
    <footer style="text-align:center; font-size:small; margin-top:50px;">
    Developed by Rashmini - A Deep Learning Capstone Project  
    <br><br>
    <strong>Note:</strong> This app is for educational purposes and should not replace professional plant health diagnostics.
    </footer>
    """, unsafe_allow_html=True)
