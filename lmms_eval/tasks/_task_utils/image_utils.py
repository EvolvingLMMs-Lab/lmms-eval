from PIL import Image

_ALPHA_MODES = ("RGBA", "LA", "PA")


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert an image to RGB, compositing any transparency onto white.

    ``Image.convert("RGB")`` silently drops the alpha channel, so transparent
    pixels keep whatever colour is stored under them. In several benchmarks that
    colour is black, which turns black line art, axes and labels on a
    transparent background into an all-black image. Compositing onto a white
    background first keeps that content visible.

    Images without transparency go through the plain ``convert("RGB")`` path, so
    their pixels are unchanged.

    Args:
        image: A PIL image in any mode.

    Returns:
        A new RGB image.
    """
    if image.mode in _ALPHA_MODES or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")
