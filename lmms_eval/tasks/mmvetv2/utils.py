import os
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import requests
import yaml
from loguru import logger as eval_logger


def add_order_label(image, label, font_size=40):
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define font for the label
    # font_path = fm.findfont(fm.FontProperties(family=font_family))
    font_path = os.path.join(__file__, os.pardir, "arial.ttf")
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text size and position
    text_width = text_height = font_size
    label_background_margin = 10
    label_background_size = (
        text_width + 2 * label_background_margin,
        text_height + 2 * label_background_margin,
    )

    # Draw a solid white square for the label background
    label_background_position = (0, 0)  # Top-left corner
    draw.rectangle(
        [
            label_background_position,
            (
                label_background_position[0] + label_background_size[0],
                label_background_position[1] + label_background_size[1],
            ),
        ],
        fill="white",
    )

    # Add the label text in black over the white square
    label_position = (label_background_margin, label_background_margin)
    draw.text(label_position, label, font=font, fill="black")

    return image


def resize_image_height(image, fixed_size):
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = int(width * fixed_size / height), fixed_size
    return image.resize(new_size, Image.Resampling.LANCZOS)


def concatenate_images_horizontal(image_list):
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    assert all(height == max_height for height in heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in image_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def resize_image_width(image, fixed_size):
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = fixed_size, int(height * fixed_size / width)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def concatenate_images_vertical(image_list):
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list))
    total_height = sum(heights)
    max_width = max(widths)
    assert all(width == max_width for width in widths)
    new_im = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for im in image_list:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def process_images_horizontal(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        # Resize image
        img_resized = resize_image_height(img, fixed_size=size)

        # Add order label
        img_labeled = add_order_label(img_resized, f"[{i+1}]")

        # Append to list
        images.append(img_labeled)

    # Concatenate all images
    return concatenate_images_horizontal(images)


def process_images_vertical(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        # Resize image
        img_resized = resize_image_width(img, fixed_size=size)

        # Add order label
        img_labeled = add_order_label(img_resized, f"[{i+1}]")

        # Append to list
        images.append(img_labeled)

    # Concatenate all images
    return concatenate_images_vertical(images)


def process_images(images, size=1008):
    concat_horizontal = process_images_horizontal(images, size)
    concat_vertical = process_images_vertical(images, size)

    hw, hh = concat_horizontal.size
    vw, vh = concat_vertical.size

    ha = hw / hh
    va = vh / vw

    if ha > va:
        return concat_vertical
    else:
        return concat_horizontal


#


def get_images_tokens(input_string):
    images = []
    queries = input_string.split("<IMG>")
    for query in queries:
        query = query.strip()
        if query.endswith((".jpg", ".png", ".jpeg")):
            # image_path = os.path.join(image_folder, query)
            # images.append(Image.open(image_path).convert("RGB"))
            images.append(query)
    return images


def mmvet_group_img_doc_to_visual(doc):
    # if doc["image"] is None:
    #     return []
    prompt = doc["question"]
    image_tokens = get_images_tokens(prompt)
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    visual = process_images(visual)
    return [visual]


def mmvet_doc_to_visual(doc):
    # if doc["image"] is None:
    #     return []
    prompt = doc["question"]
    image_tokens = get_images_tokens(prompt)
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


# def mmvet_doc_to_visual(doc):
#     if doc["image"] is None:
#         return []
#     return [doc["image"].convert("RGB")]


def replace_images_tokens(input_string):
    text_queries = []
    queries = input_string.split("<IMG>")
    for query in queries:
        query = query.strip()
        if query.endswith((".jpg", ".png", ".jpeg")):
            text_queries.append("[<IMG_PLH>]")
        else:
            text_queries.append(query)
    question = "".join(text_queries)
    return question


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        return replace_images_tokens(doc["question"])
    question = replace_images_tokens(doc["question"])
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{post_prompt}"


with open(Path(__file__).parent / "mmvetv2.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
MM_VET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.
gpt_query_prompt | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""


def get_chat_response(
    prompt,
    model=GPT_EVAL_MODEL_NAME,
    temperature=0.0,
    max_tokens=128,
    patience=3,
    sleep_time=5,
):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if API_TYPE == "azure":
        payload.pop("model")

    while patience > 0:
        patience -= 1
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]

        except Exception as e:
            eval_logger.error(f"Error: {e}")
            if "Rate limit" in str(e):
                eval_logger.info("Sleeping due to rate limit...")
                time.sleep(sleep_time)
            eval_logger.info(f"Retrying...Patience left: {patience}")

    return "", ""


def mmvet_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def mmvet_process_results(doc, results):
    # get pred and ground truth here
    pred = results[0]
    question = doc["question"]
    answer = doc["answer"]
    gpt_query_prompt = f"{MM_VET_PROMPT}\n{question} | {answer.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> ')} | {pred} |"
    grade_sample_run_complete = False
    temperature = 0.0

    while not grade_sample_run_complete:
        content, model_name = get_chat_response(
            gpt_query_prompt,
            temperature=temperature,
        )
        if content:
            try:
                content = content.split(" ")[0].strip()
                score = float(content)
                if 0.0 <= score <= 1.0:
                    grade_sample_run_complete = True
            except ValueError:
                time.sleep(5)
                temperature += 0.5
                eval_logger.info(f"Sleep 5 secs, {doc['question_id']} try again with increased temperature {temperature}.")
                content, model_name = get_chat_response(
                    gpt_query_prompt,
                    temperature=temperature,
                )
                if temperature >= 2:  # Assuming a max temperature threshold
                    score = 0.0
                    grade_sample_run_complete = True
                    eval_logger.info(f"Reach to max trials, {doc['question_id']} failed to get a score.")
        else:
            score = 0.0
            grade_sample_run_complete = True
            eval_logger.info(f"{doc['question_id']} failed to get a score.")

    return {
        f"gpt_eval_score": {
            "question_id": doc["question_id"],
            "question": doc["question"],
            "gt_answer": doc["answer"],
            "capabilities": doc["capability"],
            "pred_answer": pred,
            "score": score,
            "eval_model": model_name,
        }
    }


cap_columns = pd.DataFrame(["rec", "ocr", "know", "gen", "spat", "math", "seq"])
cap_details_columns = pd.DataFrame(
    [
        "rec_know_gen",
        "rec",
        "rec_spat_ocr_gen",
        "rec_gen",
        "ocr",
        "rec_spat",
        "spat_ocr",
        "rec_spat_gen",
        "math_spat_ocr",
        "rec_seq_gen",
        "ocr_gen",
        "rec_know",
        "rec_know_ocr_gen",
        "rec_spat_ocr",
        "math_ocr",
        "rec_know_spat",
        "rec_spat_seq_gen",
        "rec_spat_seq",
        "rec_seq_ocr_gen",
        "rec_seq",
        "rec_ocr_gen",
        "rec_ocr",
        "rec_know_spat_ocr",
        "know_spat_ocr",
        "seq_ocr_gen_rec_spat",
        "rec_seq_spat_ocr",
        "rec_know_spat_gen",
        "spat_ocr_gen",
        "rec_know_math",
        "ocr_gen_rec_know_spat",
        "rec_know_seq_gen",
        "rec_math_spat_ocr",
        "seq_ocr_gen_rec_know",
        "math_spat_ocr_gen",
        "rec_math_ocr",
        "spat_seq_ocr_rec_math",
        "spat_seq_ocr_gen_math",
        "rec_know_seq",
        "rec_seq_ocr",
    ]
)


def mmvet_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    # Calculate the overall score
    overall_score = sum([result["score"] for result in results]) / len(results) * 100
    eval_logger.info(f"Overall Score: {overall_score:.2f}")

    # Initialize dictionaries to store scores for each capability and detail
    cap_scores = {cap: 0 for cap in cap_columns.squeeze().tolist()}
    cap_details_scores = {detail: 0 for detail in cap_details_columns.squeeze().tolist()}

    # Count the number of results for each capability and detail
    cap_counts = {cap: 0 for cap in cap_scores}
    cap_details_counts = {detail: 0 for detail in cap_details_scores}

    # Aggregate scores for each capability and detail
    for result in results:
        for cap in cap_scores:
            if cap in result["capabilities"]:
                cap_scores[cap] += result["score"]
                cap_counts[cap] += 1
        for detail in cap_details_scores:
            detail_set = set(detail.split("_"))
            result_detail_set = set(result["capabilities"].split(","))
            if detail_set == result_detail_set:
                cap_details_scores[detail] += result["score"]
                cap_details_counts[detail] += 1

    # Calculate the average score for each capability
    for cap in cap_scores:
        if cap_counts[cap] > 0:
            cap_scores[cap] = cap_scores[cap] / cap_counts[cap] * 100
        eval_logger.info(f"Score for {cap}: {cap_scores[cap]:.2f}")

    # Calculate the average score for each detailed capability
    for detail in cap_details_scores:
        if cap_details_counts[detail] > 0:
            cap_details_scores[detail] = cap_details_scores[detail] / cap_details_counts[detail] * 100
        eval_logger.info(f"Score for {detail}: {cap_details_scores[detail]:.2f}")

    return overall_score
