import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model('cifar_cnn_model.h5')

class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]


st.title('CIFAR-10 Image Classification')


uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    img_resized = img.resize((32, 32))
    img_array = image.img_to_array(img_resized) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Class: {class_names[predicted_class[0]]}')
    st.write(f'Prediction Score: {np.max(predictions) * 100:.2f}%')
