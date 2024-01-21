import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import inception_v3
from torch import nn

# Load the trained InceptionV3 model
model = inception_v3(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes
model.load_state_dict(torch.load('inception_model_state_dict.pth'))
model.eval()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("Waste Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Perform inference
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)

    # Display the result
    class_names = ['Organic', 'Recyclable']
    _, predicted_class = output.max(1)
    result = class_names[predicted_class.item()]
    
    st.write(f"Prediction: {result}")
