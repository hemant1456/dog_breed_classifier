import streamlit as st
import hydra
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import LitModel
import io
import os

def set_page_config():
    st.set_page_config(
        page_title="Dog Breed Predictor",
        page_icon="🐶",
        layout="wide",
    )

def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to right top, #d16ba5, #c777b9, #ba83ca, #aa8fd8, #9a9ae1, #8aa7ec, #79b3f4, #69bff8, #52cffe, #41dfff, #46eefa, #5ffbf1);
        background-attachment: fixed;
    }
    .main-header {
        font-family: 'Trebuchet MS', sans-serif;
        color: #1E1E1E;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.7);
        margin-bottom: 30px;
    }
    .prediction-result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        margin-top: 20px;
    }
    .uploaded-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

@hydra.main(version_base=None, config_path="configs", config_name="configs.yaml")
def app(cfg):
    set_page_config()
    load_css()
    
    st.markdown("<h1 class='main-header'>🐶 Dog Breed Prediction 🐾</h1>", unsafe_allow_html=True)
    
    # Load model
    @st.cache_resource
    def load_model():
        lit_model = LitModel.load_from_checkpoint(cfg.best_checkpoint_path, cfg=cfg)
        lit_model.eval()
        return lit_model
    
    lit_model = load_model()
    
    # Get class names from the dataset folder
    class_names = sorted(os.listdir(cfg.folder.dataset))
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "png", "jpeg"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Read the file and convert it to a PIL Image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize the image to 100x100
        image_display = image.copy()
        image_display.thumbnail((400, 400))
        
        # Display the uploaded image
        with col1:
            st.image(image_display, caption="Uploaded Image", use_column_width=False, output_format="PNG", width=400)
        
        # Predict button
        with col2:
            if st.button("🔍 Predict Breed", key="predict_button"):
                with st.spinner("Analyzing the image..."):
                    # Preprocess the image
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = preprocess(image)
                    input_tensor = input_tensor.to(cfg.trainer.accelerator)
                    input_batch = input_tensor.unsqueeze(0)
                    
                    # Prediction
                    with torch.no_grad():
                        output = lit_model(input_batch)
                    
                    # Get the predicted class and probability
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    top_prob, top_catid = torch.topk(probabilities, 1)
                    
                    # Map the predicted index to class name
                    predicted_breed = class_names[top_catid.item()]
                    confidence = top_prob.item() * 100
                    
                    st.markdown(f"<div class='prediction-result'>Predicted Dog Breed:<br>{predicted_breed}<br>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
    
    st.sidebar.header("About")
    st.sidebar.info("This app uses a deep learning model to predict dog breeds from uploaded images. Upload a clear image of a dog to get started!")
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload a clear image of a dog
    2. Click the 'Predict Breed' button
    3. View the predicted breed and confidence level
    """)

if __name__ == "__main__":
    app()