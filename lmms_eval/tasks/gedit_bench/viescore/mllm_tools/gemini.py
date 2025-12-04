"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import os
import tempfile
from io import BytesIO
from typing import List
from urllib.parse import urlparse

import google.generativeai as genai
import requests
from PIL import Image

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def upload_to_gemini(input, mime_type=None):
    """Uploads the given file or PIL image to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    if isinstance(input, str):
        # Input is a file path
        file = genai.upload_file(input, mime_type=mime_type)
    elif isinstance(input, Image.Image):
        # Input is a PIL image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            input.save(tmp_file, format="JPEG")
            tmp_file_path = tmp_file.name
        file = genai.upload_file(tmp_file_path, mime_type=mime_type or "image/jpeg")
        os.remove(tmp_file_path)
    else:
        raise ValueError("Unsupported input type. Must be a file path or PIL Image.")

    # print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def save_image_from_url(url, base_save_directory="tmp", file_name=None):
    # Parse the URL to create a directory path
    parsed_url = urlparse(url)
    url_path = os.path.join(parsed_url.netloc, parsed_url.path.lstrip("/"))
    save_directory = os.path.join(base_save_directory, os.path.dirname(url_path))

    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Get the image from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Open the image
        image = Image.open(BytesIO(response.content))

        # Set the file name if not provided
        if not file_name:
            file_name = os.path.basename(parsed_url.path)

        # Save the image locally
        file_path = os.path.join(save_directory, file_name)
        image.save(file_path)

        return file_path
    else:
        raise Exception(f"Failed to retrieve image from URL. Status code: {response.status_code}")


class Gemini:
    def __init__(self, model_name="gemini-1.5-pro-latest"):
        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]

        images_prompt = []
        for image_link in image_links:
            if isinstance(image_link, str):
                image = save_image_from_url(image_link)
            else:
                image = image_link
            image = upload_to_gemini(image, mime_type="image/jpeg")
            images_prompt.append(image)

        prompt_content = [images_prompt, text_prompt]
        return prompt_content

    def get_parsed_output(self, prompt):
        images_prompt = prompt[0]
        text_prompt = prompt[1]
        chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": images_prompt,
                },
            ]
        )
        try:
            response = chat_session.send_message(text_prompt)
        except:
            return "Error in sending message to chat session."
        return self.extract_response(response)

    def extract_response(self, response):
        response = response.text
        return response


if __name__ == "__main__":
    model = Gemini()
    prompt = model.prepare_prompt(
        ["https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg", "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"], "What is difference between two images?"
    )
    print("prompt : \n", prompt)
    res = model.get_parsed_output(prompt)
    print("result : \n", res)
