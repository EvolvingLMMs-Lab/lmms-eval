# server.py
import base64
import io
from typing import List

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image

app = FastMCP("demo")


@app.tool(name="image_zoom_in_tool", description="Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.")
def image_zoom_in_tool(image_path: str, bbox: List[float]):
    """
    Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.

    :param image_path: Path to the input image.
    :param bbox: Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    :return: Cropped image as a base64-encoded string.
    """
    image = Image.open(image_path)

    cropped_image = image.crop(bbox)

    image_bytes = io.BytesIO()
    cropped_image.save(image_bytes, format="PNG")

    image_bytes.seek(0)
    png = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    return ImageContent(type="image", data=png, mimeType="image/png")


@app.tool(name="weather", description="query weather")
def get_weather(city: str):
    weather_data = {"Beijing": {"temp": 25, "condition": "Rainy"}, "Shanghai": {"temp": 28, "condition": "Cloudy"}}
    # 返回对应城市的天气信息，如果城市不存在则返回错误信息
    result = weather_data.get(city, {"error": "未找到该城市"})
    return result


@app.tool(name="get_blank_image", description="get blank image")
def get_blank_image(width: int = 512, height: int = 512):
    """ """
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    png = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return ImageContent(type="image", data=png, mimeType="image/png")


if __name__ == "__main__":
    app.run()
