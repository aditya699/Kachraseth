import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pywhatkit
from datetime import datetime

# Azure Custom Vision API endpoint and key
api_endpoint = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/764d68b7-5cdd-4379-a979-d6f2178c0ef6/classify/iterations/Iteration1/image"
api_key = "48e3917cdedb48868e1c0f0b7786a8e2"

# Set Streamlit app title
st.title("Kachraseth")

# Ask the user for an address
user_address = st.text_input("Enter the address where waste is found:")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert PIL image to bytes
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")

    # Send the image to Azure Custom Vision for prediction
    headers = {
        "Prediction-Key": api_key,
        "Content-Type": "application/octet-stream"
    }

    # Reset the stream position to the beginning
    image_bytes.seek(0)

    # Send the request
    response = requests.post(api_endpoint, headers=headers, data=image_bytes)

    # Display prediction results
    if response.status_code == 200:
        prediction_result = response.json()
        predictions = prediction_result["predictions"]

        # Find the class with the highest probability
        highest_prediction = max(predictions, key=lambda x: x['probability'])

        # Display the result based on the class
        if highest_prediction['tagName'] == 'o':
            st.write("Prediction: The Following waste is classified as Organic")
            current_time = datetime.now().strftime("%H:%M")
            pywhatkit.sendwhatmsg("+917303041453", 
                                  f"Organic Waste is found at {user_address} at {current_time}!", 
                                  19, 40)
        elif highest_prediction['tagName'] == 'r':
            st.write("Prediction: The Following waste is classified as Recyclable")
            current_time = datetime.now().strftime("%H:%M")
            pywhatkit.sendwhatmsg("+917303041453",
                                  f"Recyclable waste is found at {user_address} at {current_time}!", 
                                  19, 40)
    else:
        st.write(f"Error: {response.status_code} - {response.text}")
