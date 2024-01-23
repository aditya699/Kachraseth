import requests

# Replace <image file> with the actual path to your image file
image_path = "C:/Users/aditya/Desktop/2024/Kachraseth/Lenet-5-POC/Data/train/R/R_9216.jpg"

# API endpoint and key
api_endpoint = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/764d68b7-5cdd-4379-a979-d6f2178c0ef6/classify/iterations/Iteration1/image"
api_key = "48e3917cdedb48868e1c0f0b7786a8e2"

# Set headers
headers = {
    "Prediction-Key": api_key,
    "Content-Type": "application/octet-stream"
}

# Read the image file
with open(image_path, "rb") as image_file:
    # Send the request
    response = requests.post(api_endpoint, headers=headers, data=image_file)

# Print the response
print(response.json())

