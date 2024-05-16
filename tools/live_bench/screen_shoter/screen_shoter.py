from selenium import webdriver
from PIL import Image
from lmms_eval.live_bench.websites import Website
from lmms_eval.live_bench.screen_shoter.screen import ScreenImage
from typing import List
from abc import ABC, abstractmethod
from datetime import datetime
from PIL import Image
import os
import io
import logging

logger = logging.getLogger("lmms-eval")


class ScreenShoter(ABC):
    def __init__(self, screen_size=(1024, 1024)):
        self.screen_size = screen_size

    def capture(self, driver: webdriver.Chrome, website: Website) -> ScreenImage:
        if driver is not None:
            website.visit(driver)
            if self.screen_size != "auto":
                driver.set_window_size(self.screen_size[0], self.screen_size[1])
            else:
                driver.set_window_size(1024, 1024)
                page_width = driver.execute_script("return document.body.scrollWidth")
                driver.set_window_size(page_width, 1024)
        # print("Screen size:", driver.get_window_size())
        images = self.get_screenshot(driver)
        return ScreenImage(images, website, self.get_name(), self.screen_size, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def __call__(self, driver: webdriver.Chrome, website: Website) -> List[Image.Image]:
        return self.capture(driver, website)

    def get_name(self) -> str:
        raise NotImplementedError("get_name not implemented")

    @abstractmethod
    def get_screenshot(self, driver: webdriver.Chrome) -> List[Image.Image]:
        pass


class ScreenShoterRegistry:
    def __init__(self):
        self.shoters = {}

    def register_shoter(self, name):
        def decorator(cls):
            self.shoters[name] = cls
            cls.get_name = lambda self: name
            return cls

        return decorator

    def get_shoter(self, name) -> ScreenShoter:
        return self.shoters[name]


shoter_registry = ScreenShoterRegistry()


def register_shoter(name):
    return shoter_registry.register_shoter(name)


def get_shoter(name, *args, **kwargs) -> ScreenShoter:
    return shoter_registry.get_shoter(name)(*args, **kwargs)


@register_shoter("human")
class HumanScreenShoter(ScreenShoter):
    def __init__(self, screen_size=None):
        super().__init__(screen_size)

    def capture(self, driver: webdriver.Chrome, website: Website) -> ScreenImage:
        path = website.get_path()
        images = []

        def get_image(path):
            try:
                with open(path, "rb") as f:
                    image_data = f.read()
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")

        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file_name in files:
                    get_image(os.path.join(root, file_name))
        else:
            try:
                get_image(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
        if not images:
            raise ValueError(f"No images found in {path}")
        return ScreenImage(images, website, self.get_name(), self.screen_size, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_screenshot(self, driver: webdriver.Chrome) -> List[Image.Image]:
        return []


@register_shoter("single_screen")
class SingleScreenShoter(ScreenShoter):
    def __init__(self, screen_size=(1024, 1024)):
        super().__init__(screen_size)

    def get_screenshot(self, driver: webdriver.Chrome) -> List[Image.Image]:
        screenshot = driver.get_screenshot_as_png()
        return [Image.open(io.BytesIO(screenshot))]


@register_shoter("rolling_screen")
class RollingScreenShoter(ScreenShoter):
    def __init__(self, screen_size=(1024, 1024)):
        super().__init__(screen_size)

    def get_screenshot(self, driver: webdriver.Chrome) -> List[Image.Image]:
        screenshots = []
        # Scroll to the top of the page before taking the first screenshot
        driver.execute_script("window.scrollTo(0, 0)")
        # Get the total height of the web page
        total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        # Get the viewport height
        viewport_height = driver.execute_script("return window.innerHeight")
        # Initialize the current scroll position
        current_scroll_position = 0

        # Scroll through the page and take screenshots
        while current_scroll_position < total_height:
            # Take screenshot and append to the list
            screenshot = driver.get_screenshot_as_png()
            screenshots.append(Image.open(io.BytesIO(screenshot)))
            # Scroll down by the viewport height
            current_scroll_position += viewport_height
            driver.execute_script(f"window.scrollTo(0, {current_scroll_position})")

        return screenshots
