import base64
import json
from io import BytesIO

import requests
from PIL import Image

# Test URL for comparison
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


# Function to encode image to base64
def encode_image_to_base64(image_url):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # Resize if needed (optional)
    # img = img.resize((800, 600))

    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Encode the image
base64_image = encode_image_to_base64(image_url)

# PART 1: Test with URL-based image
print("TESTING URL-BASED IMAGE:")
response_url = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer ",
        "Content-Type": "application/json",
    },
    data=json.dumps(
        {
            "model": "meta-llama/llama-4-maverick:free",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "what is in this image?"}, {"type": "image_url", "image_url": {"url": image_url}}]}],
        }
    ),
)

# Print URL-based response
print("URL-based image status code:", response_url.status_code)
if response_url.status_code == 200:
    print("URL-based image response:", response_url.json()["choices"][0]["message"]["content"])
else:
    print("URL-based image error:", response_url.text)

# PART 2: Test with base64-encoded image
print("\nTESTING BASE64-ENCODED IMAGE:")
response_base64 = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer ",
        "Content-Type": "application/json",
    },
    data=json.dumps(
        {
            "model": "meta-llama/llama-4-maverick:free",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "what is in this image?"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}],
        }
    ),
)

# Print base64-encoded response
print("Base64-encoded image status code:", response_base64.status_code)
if response_base64.status_code == 200:
    print("Base64-encoded image response:", response_base64.json()["choices"][0]["message"]["content"])
else:
    print("Base64-encoded image error:", response_base64.text)

# Compare responses
if response_url.status_code == 200 and response_base64.status_code == 200:
    url_response = response_url.json()["choices"][0]["message"]["content"]
    base64_response = response_base64.json()["choices"][0]["message"]["content"]
    print("\nCOMPARISON:")
    print("Are responses identical?", url_response == base64_response)
    if url_response != base64_response:
        print("Responses differ, but this is expected as the model may have some non-deterministic behavior")
