import streamlit as st
from utils import set_background
from PIL import Image
from tensorflow import keras
from classifier import classifier
from io import BytesIO
import base64

st.set_page_config(
    page_title='Brain Tumor Classification',
    layout='centered'
)

set_background('utils/bg.jpg')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 60px;
        color: #f0f0f0;
        font-weight: bold;        
        margin-top: -75px;
    }
    .header {
        display: flex;
        justify-content: center;  /* Center horizontally */
        align-items: center;  /* Center vertically (if needed) */
        text-align: center;  
        font-size: 30px;
        color: #87cefa;
        white-space: nowrap;
        margin-top: -20px;
        width: 100%;  /* Ensures full width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Brain Tumor Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify it as Brain Tumor or No Brain Tumor.</div>', unsafe_allow_html=True)

file = st.file_uploader('',type = ['jpg','jpeg','png','jfif'])

model = keras.models.load_model("Model/model.keras" )

class_names = {0:'No Brain Tumor', 1:'Brain Tumor'}

if file is not None:
    
    image = Image.open(file).convert('RGB')

    prediction, score = classifier(image, model, class_names)
   
    bufferd = BytesIO()
    image.save(bufferd, format='PNG')
    img_base64 = base64.b64encode(bufferd.getvalue()).decode()

    # Display classification results with reduced gap and no extra space
    st.markdown(
        f"""
        <div style="align-items: center;">        
        <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover; margin-top: -20px;"/>
        <div style="font-size:40px; font-weight:bold; margin-left: 20px; color:#FFFFFF; white-space: nowrap;">
            <p> <strong>Result: {prediction}</strong></p>
            <p style="margin-top:-10px;"> <strong> Score: {score}% </strong> </p>
        </div>        
        </div>
        """,
        unsafe_allow_html=True
    )