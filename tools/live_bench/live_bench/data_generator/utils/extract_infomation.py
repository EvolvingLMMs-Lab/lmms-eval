import os
import openai
import logging
from bs4 import BeautifulSoup
import requests
from live_bench.data_generator.utils.gpt4v import gpt4v_generate_response, format_gpt4v_images
from live_bench.screen_shoter import ScreenImage
from live_bench.websites import Website
from live_bench.data_generator.response import Response

logger = logging.getLogger("live-bench")


GPT4V_EXTRACT_TEXT_PROMPT: str = """\
These are the images of the website that we have captured. Please extract the text from the website.
You should extract the text from the website as detailed as possible.
Only output the text extracted from the website, do not include any other information.
"""

GPT4V_FIND_IMAGES_FEATURES_PROMPT: str = """\
This is a screenshot from a news website. Your task is to identify the meaningful images in this screenshot and extract relevant information about these images, such as the environment depicted, the actions and expressions of the people, and the connection between these images and the corresponding text. You need to think deeply about these images and provide as much detailed and useful information as possible.
"""

GPT4V_THINK_DIFFERENTLY_PROMPT: str = """\
What makes this website different from other websites? What is special about its news? Since it is a news website, where is the "new" reflected? Do not give a generalized answer; you need to provide detailed answers based on the specific content of each news article and the accompanying illustrations.
"""


class ImageInfomation(object):
    def __init__(self, text=None, image_features=None, differnt_points=None):
        self.text = text
        self.image_features = image_features
        self.differnt_points = differnt_points

    def to_dict(self):
        res = {}
        if self.text:
            res["Text Extracted in the HTML"] = self.text
        if self.image_features:
            res["Image Features"] = self.image_features
        if self.differnt_points:
            res["Interesting Points"] = self.differnt_points
        return res

    def __str__(self):
        return self.get_info()

    def get_info(self):
        res_list = [f"## {key}\n\n{value}" for key, value in self.to_dict().items()]
        if res_list:
            return "**Here is something you can take as reference.**\n\n" + "\n\n".join(res_list)
        else:
            return ""


class InfomationExtractor(object):
    def __init__(self, model="gpt-4-turbo", openai_api_key=None):
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model

    def extract_text_from_html(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        text = "\n".join(soup.stripped_strings)
        return text

    def extract_text_from_html_using_gpt4v(self, screen_image: ScreenImage, **kwargs) -> Response:
        website: Website = screen_image.website
        if website.url:
            url = website.url
            text = self.extract_text_from_html(url)
            text = f"Below is the text extracted from the website {url} for you to take reference:\n{text}"
        else:
            text = ""
        text = f"{GPT4V_EXTRACT_TEXT_PROMPT}\n{text}"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}] + format_gpt4v_images(screen_image.images),
            }
        ]
        response = gpt4v_generate_response(messages, model=self.model, client=self.client, json_format=False, **kwargs)
        return response

    def extract_infomation(self, screen_image: ScreenImage, **kwargs) -> ImageInfomation:
        ocrs = self.extract_text_from_html_using_gpt4v(screen_image)
        infomation = ImageInfomation()
        if ocrs.success:
            ocrs = f"Below is the text extracted from the website for you to take reference:\n{ocrs.content}"
            infomation.text = ocrs
        else:
            ocrs = ""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{GPT4V_FIND_IMAGES_FEATURES_PROMPT}\n{ocrs}"}] + format_gpt4v_images(screen_image.images),
            }
        ]
        response = gpt4v_generate_response(messages, model=self.model, client=self.client, json_format=False, **kwargs)
        if response.success:
            infomation.image_features = response.content
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{GPT4V_THINK_DIFFERENTLY_PROMPT}\n\n{str(infomation)}"}] + format_gpt4v_images(screen_image.images),
            }
        ]
        response = gpt4v_generate_response(messages, model=self.model, client=self.client, json_format=False, **kwargs)
        if response.success:
            infomation.differnt_points = response.content
        return infomation
