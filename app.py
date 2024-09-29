import streamlit as st
import hydra
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import LitModel
import io
import os

@hydra.main(version_base=None, config_path="configs", config_name="configs.yaml")
def app(cfg):
    st.title("Dog Breed Prediction")
    
    # Load model
    lit_model = LitModel.load_from_checkpoint(cfg.best_checkpoint_path, cfg=cfg)
    lit_model.eval()
    
    # Get class names from the dataset folder
    class_names = sorted(os.listdir(cfg.folder.dataset))
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Read the file and convert it to a PIL Image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            # Preprocess the image
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image)
            # change the image to the acclerator mention in the config
            input_tensor = input_tensor.to(cfg.trainer.accelerator)
            input_batch = input_tensor.unsqueeze(0)

            
            # Prediction
            with torch.no_grad():
                output = lit_model(input_batch)
            
            # Get the predicted class
            _, predicted_idx = torch.max(output, 1)
            
            # Map the predicted index to class name
            predicted_breed = class_names[predicted_idx.item()]
            
            st.success(f"Predicted Dog Breed: {predicted_breed}")

if __name__ == "__main__":
    app()