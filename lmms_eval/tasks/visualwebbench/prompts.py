WEB_CAPTION_PROMPT = """You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name="description" content=""> HTML element.

You should use the following format, and do not output any explanation or any other contents:
<meta name="description" content="YOUR ANSWER">
"""

HEADING_OCR_PROMPT = """You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.

You should directly tell me the main content, and do not output any explanation or any other contents.
"""

WEBQA_PROMPT = """{question}
You should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.
"""

ELEMENT_OCR_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please perform OCR in the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

"""You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is [0.20, 0.39, 0.80, 0.41].
Please perform OCR on the bounding box and recognize the text content within the red bounding box.

You should use the following format:
The text content within the red bounding box is: <YOUR ANSWER>
"""

"output_v7/long_text_OCR/annotated_images/iheartdogs.com_start6864_annotated53.png"

ELEMENT_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one best matches the description: {element_desc}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_PREDICTION_PROMPT = """You are given a screenshot of a webpage with a red rectangle bounding box. The [x1, y1, x2, y2] coordinates of the bounding box is {bbox_ratio}.
Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box:
{choices_text}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""

ACTION_GROUND_PROMPT = """In this website screenshot, I have labeled IDs for some HTML elements as candicates. Tell me which one I should click to complete the following task: {instruction}

You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.
"""


DEFAULT_PROMPTS = {
    "web_caption": WEB_CAPTION_PROMPT,
    "heading_ocr": HEADING_OCR_PROMPT,
    "webqa": WEBQA_PROMPT,
    "element_ocr": ELEMENT_OCR_PROMPT,
    "element_ground": ELEMENT_GROUND_PROMPT,
    "action_prediction": ACTION_PREDICTION_PROMPT,
    "action_ground": ACTION_GROUND_PROMPT,
}
