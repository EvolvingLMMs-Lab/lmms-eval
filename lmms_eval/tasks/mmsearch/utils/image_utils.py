import os
import shutil
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000
from loguru import logger as eval_logger


# slim fullpage screenshot
def slim_image_and_save(image_path, save_path):
    result, is_gray = adaptive_pixel_slimming(image_path)
    try:
        if is_gray:
            save_result = cv2.imwrite(save_path, result.reshape(-1, 1))
        else:
            save_result = cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    except Exception as e:
        eval_logger.info(f"Slim fullpage screenshot failed: {e}")
        shutil.copy(image_path, save_path)
        return

    if not save_result:
        eval_logger.info("Save slimmed fullpage screenshot failed")
        shutil.copy(image_path, save_path)


def adaptive_pixel_slimming(image_path, RESIZE_W=1024, RESIZE_H=5120, thresh_gradmap=200, thresh_gradsum=50, thresh_length=15):
    # Read the source document image
    ori_website = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if ori_website is None:
        eval_logger.warning("cv2 load failed. Using PIL to load")
        ori_website_rgb = Image.open(image_path).convert("RGB")
        ori_website_rgb = np.asanyarray(ori_website_rgb)
        ori_website = ori_website_rgb[:, :, [2, 1, 0]]

    # Check if the image is grayscale or color
    is_gray = len(ori_website.shape) == 2 or (len(ori_website.shape) == 3 and ori_website.shape[2] == 1)

    if is_gray:
        ori_website = ori_website.squeeze() if len(ori_website.shape) == 3 else ori_website
    else:
        ori_website = cv2.cvtColor(ori_website, cv2.COLOR_BGR2RGB)

    H, W = ori_website.shape[:2]

    # Compute Sobel gradients
    if is_gray:
        sobel_x = np.abs(cv2.Sobel(ori_website, cv2.CV_64F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(ori_website, cv2.CV_64F, 0, 1, ksize=3))
    else:
        sobel_x = np.abs(cv2.Sobel(ori_website, cv2.CV_64F, 1, 0, ksize=3)).max(axis=2)
        sobel_y = np.abs(cv2.Sobel(ori_website, cv2.CV_64F, 0, 1, ksize=3)).max(axis=2)

    # Compute gradient map
    ori_website_gradient_map = np.maximum(sobel_x, sobel_y)
    # Resize to apply the threshold
    ori_website_gradient_map = cv2.resize(ori_website_gradient_map, (RESIZE_W, RESIZE_H))
    ori_website_gradient_map[ori_website_gradient_map < thresh_gradmap] = 0

    # Find blank area in y direction
    sum_grad_y = np.sum(ori_website_gradient_map, axis=0)
    blank_blocks_y = find_blank_block(sum_grad_y, thresh_gradsum, thresh_length)
    blank_blocks_y = resize2predefined(blank_blocks_y, W / RESIZE_W)

    # Find blank area in x direction
    sum_grad_x = np.sum(ori_website_gradient_map, axis=1)
    blank_blocks_x = find_blank_block(sum_grad_x, thresh_gradsum, thresh_length)
    blank_blocks_x = resize2predefined(blank_blocks_x, H / RESIZE_H)
    # Remove blank blocks
    slimmed_website = remove_blocks(blank_blocks_y, blank_blocks_x, ori_website)

    return slimmed_website, is_gray


def find_blank_block(arr, thresh_gradsum, thresh_length):
    mask = (arr > thresh_gradsum).astype(int)
    diff = np.diff(np.concatenate(([1], mask, [1])))
    end_indices = np.where(diff == 1)[0]
    start_indices = np.where(diff == -1)[0]
    lengths = end_indices - start_indices
    valid_blocks = lengths >= thresh_length
    return list(zip(start_indices[valid_blocks], end_indices[valid_blocks]))


def resize2predefined(blocks, scale):
    return [(int(start * scale), int(end * scale)) for start, end in blocks]


def remove_blocks(blank_blocks_y, blank_blocks_x, image):
    mask = np.ones(image.shape[:2], dtype=bool)
    for start, end in blank_blocks_y:
        mask[:, start:end] = False
    for start, end in blank_blocks_x:
        mask[start:end, :] = False

    # Extract non-black regions
    if len(image.shape) == 2:  # Grayscale
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        return image[rows][:, cols]
    else:  # Color
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        return image[rows][:, cols]


# crop & split fullpage screenshot to multiple small images
def crop_and_split(fullpage_path, fullpage_split_dict, save_slice_path=None):
    slice_height = fullpage_split_dict["slice_height"]
    max_slices = fullpage_split_dict["max_slices"]

    return_list = []
    # Open the image
    with Image.open(fullpage_path) as img:
        width, height = img.size

        # Calculate the number of slices needed
        num_slices = min(max_slices, (height + slice_height - 1) // slice_height)

        # Slice and save the image
        for i in range(num_slices):
            top = i * slice_height
            bottom = min((i + 1) * slice_height, height)

            slice = img.crop((0, top, width, bottom))

            # Save the slice
            if save_slice_path is not None:
                output_path = os.path.join(save_slice_path, f"slice_{i}.jpg")
                slice.save(output_path)
            else:
                output_path = pil_image_to_bytes(slice)
            return_list.append(output_path)

    return return_list


# crop image search result
def crop_image_search_results(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("cv2 load failed. Using PIL to load")
        print(f"image_path: {image_path}; exist: {os.path.exists(image_path)}")
        image_rgb = Image.open(image_path)
        image_rgb = np.asanyarray(image_rgb)
        image = image_rgb[:, :, [2, 1, 0]]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply vertical Sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

    # Convert to 8-bit unsigned integer
    sobelx = np.uint8(np.absolute(sobelx))

    # Apply thresholding
    _, thresh = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(thresh, 1, np.pi / 180, 600)

    max_x = 0

    # Check detected lines
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # Only consider nearly vertical lines (theta close to 0 or pi)
            if theta < 0.1 or theta > np.pi - 0.1:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                # Calculate two endpoints of the line
                x1 = int(x0 + 1000 * (-b))
                x2 = int(x0 - 1000 * (-b))
                # Update maximum x value
                max_x = max(max_x, x1, x2)

    # Ensure max_x does not exceed image width
    max_x = min(max_x, image.shape[1] - 1)

    # Crop the image, keeping only the part to the right of max_x
    cropped_image = image[:, max_x:]
    cv2.imwrite(save_path, cropped_image)


# convert pil images to bytes to unify the loading method
# the object returned by the function can be loaded with Image.open function
def pil_image_to_bytes(pil_image, format="PNG"):
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format=format)
    return BytesIO(img_byte_arr.getvalue())


if __name__ == "__main__":
    image_path = "temp_files/20240824_163026/21/stage3/0/fullpage.png"
    if cv2.imread(image_path, cv2.IMREAD_UNCHANGED) is None:
        print("Wrong")
    else:
        print(cv2.imread(image_path, cv2.IMREAD_UNCHANGED).shape)
