import logging
import os

import anthropic
import openai
import requests
from bs4 import BeautifulSoup
from live_bench.data_generator.response import Response
from live_bench.data_generator.utils.claude import (
    claude_generate_response,
    format_claude_images,
)
from live_bench.data_generator.utils.gpt4v import (
    format_gpt4v_images,
    get_openai_client,
    gpt4v_generate_response,
)
from live_bench.screen_shoter import ScreenImage
from live_bench.websites import Website

logger = logging.getLogger("live-bench")


EXTRACT_TEXT_PROMPT: str = """\
These are the images of the website that we have captured. Please extract the text from the website.
You should extract the text from the website as detailed as possible.
Only output the text extracted from the website, do not include any other information.
"""

FIND_IMAGES_FEATURES_PROMPT: str = """\
This is a screenshot from a news website. Your task is to identify the meaningful images in this screenshot and extract relevant information about these images, such as the environment depicted, the actions and expressions of the people, and the connection between these images and the corresponding text. You need to think deeply about these images and provide as much detailed and useful information as possible. Of course, it is also possible that the website is mainly text-based, and in this case, there might not be much information to extract from the images. In such instances, you can approach the task from a textual perspective, analyzing the website's content. For example, what is the theme of the website? What kind of information is contained in the text? If the website requires thoughtful analysis, feel free to engage in deeper reflection and provide your insights.
"""

THINK_DIFFERENTLY_PROMPT: str = """\
What makes this website different from other websites? What is special about its news? Since it is a news website, where is the 'new' aspect reflected? Do not provide a generalized answer; you need to give detailed responses based on the specific content of each news article and the accompanying illustrations.

For example, if the news is about a software update, what conveniences will this update bring to people? How can people use these new features? Perhaps there are also some drawbacks? You need to come up with your own questions worth pondering about the website and describe in as much detail as possible your understanding of what is 'new' on the website.
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
    def __init__(self, model="claude-3-5-sonnet-20240620", openai_api_key=None, anthropic_api_key=None):
        if not anthropic_api_key:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY", None)
        if "gpt" in model:
            # self.client = openai.OpenAI(api_key=openai_api_key)
            self.client = get_openai_client()
            self.generate_response = gpt4v_generate_response
            self.format_images = format_gpt4v_images
        elif "claude" in model:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            self.generate_response = claude_generate_response
            self.format_images = format_claude_images
        self.model = model

    def extract_text_from_html(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = "\n".join(soup.stripped_strings)
        return text

    def extract_text_from_html_from_gpt(self, screen_image: ScreenImage, **kwargs) -> Response:
        website: Website = screen_image.website
        if website.url:
            url = website.url
            text = self.extract_text_from_html(url)
            text = f"Below is the text extracted from the website {url} for you to take reference:\n{text}"
        else:
            text = ""
        text = f"{EXTRACT_TEXT_PROMPT}\n{text}"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}] + self.format_images(screen_image.images),
            }
        ]
        response = self.generate_response(messages=messages, model=self.model, client=self.client, json_format=False, **kwargs)
        return response

    def extract_infomation(self, screen_image: ScreenImage, **kwargs) -> ImageInfomation:
        ocrs = self.extract_text_from_html_from_gpt(screen_image)
        infomation = ImageInfomation()
        if ocrs.success:
            ocrs = f"Below is the text extracted from the website for you to take reference:\n{ocrs.content}"
            infomation.text = ocrs
        else:
            ocrs = ""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{FIND_IMAGES_FEATURES_PROMPT}\n{ocrs}"}] + self.format_images(screen_image.images),
            }
        ]
        response = self.generate_response(messages=messages, model=self.model, client=self.client, json_format=False, **kwargs)
        if response.success:
            infomation.image_features = response.content
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{THINK_DIFFERENTLY_PROMPT}\n\n{str(infomation)}"}] + self.format_images(screen_image.images),
            }
        ]
        response = self.generate_response(messages=messages, model=self.model, client=self.client, json_format=False, **kwargs)
        if response.success:
            infomation.differnt_points = response.content
        return infomation
