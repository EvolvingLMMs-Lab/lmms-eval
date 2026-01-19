import random
import re
from typing import Dict, List, Tuple

import numpy as np

LETTERS = [chr(65 + i) for i in range(26)]  # A-Z


def _get_choices(doc: Dict) -> Dict[str, str]:
    lst = [str(x) for x in doc["options"]]
    n = len(lst)
    return {LETTERS[i]: lst[i] for i in range(n)}


def super_gpqa_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    q = doc["question"]
    choices = _get_choices(doc)
    lines = [q, ""]
    for letter in sorted(choices.keys()):
        text = choices[letter]
        lines.append(f"{letter}) {text}")
    question = "\n".join(lines)
    return f"{pre_prompt}{question}{post_prompt}"


def super_gpqa_doc_to_target(doc: Dict) -> str:
    return doc["answer_letter"].strip().upper()


def super_gpqa_doc_to_choice(doc: Dict) -> List[str]:
    choices = _get_choices(doc)
    return list(choices.keys())


def parse_multi_choice_response(response, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted choice letter e.g., A, B, C, D.
    """
    # Clean response of unwanted characters
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    candidates = []
    # Look for choices with parentheses, e.g., (A)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)

    # Look for simple choices, e.g., A, B, C
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Look for choices with periods, e.g., A., B., C.
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # If no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        # If more than one candidate, choose the last one found
        start_indexes = [response.rfind(f" {can} ") for can in candidates]
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index


def super_gpqa_process_results(doc: Dict, result: List[str]) -> Dict[str, float]:
    response = result[0].strip() if result else ""
    all_choices = super_gpqa_doc_to_choice(doc)
    pred = parse_multi_choice_response(response, all_choices)
    gt = super_gpqa_doc_to_target(doc)
    score = 1.0 if pred == gt else 0.0
    return {"accuracy": score}


# Few-shot multishot builder
FEWSHOT_PROMPT = (
    "Answer the following multiple-choice question. There is only one correct answer. The\n"
    "last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n"
    "Question: A refracting telescope consists of two converging lenses separated by 100 cm.\n"
    "The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope\n"
    "is ( ).\n"
    "A) 10\nB) 40\nC) 6\n69\nD) 25\nE) 15\nF) 50\nG) 30\nH) 4\nI) 5\nJ) 20\n"
    "Answer: Let's think step by step. In a refracting telescope, if both lenses are converging,\n"
    "the focus of both lenses must be between the two lenses, and thus the focal lengths of the\n"
    "two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the\n"
    "focal length of the other must be 80 cm. The magnification is the ratio of these two focal\n"
    "lengths, or 4.\n"
    "Answer: H.\n"
    "Question: Say the pupil of your eye has a diameter of 5 mm and you have a telescope\n"
    "with an aperture of 50 cm. How much more light can the telescope gather than your eye?\n"
    "A) 1000 times more\nB) 50 times more\nC) 5000 times more\nD) 500 times more\nE) 10000 times more\nF) 20000 times more\nG) 2000 times more\nH) 100 times more\nI) 10 times more\nJ) N/A\n"
    "Answer: Let's think step by step. The amount of light a telescope can gather compared to the human eye is proportional to the area of its apertures. The area of a circle is given by the formula A = pi * (D/2)^2, where D is the diameter. Therefore, the relative light-gathering power is calculated as:\n"
    "(50 cm / 0.1 cm)^2 / (5 mm / 0.1 cm)^2 = 500^2 / 5^2 = 10000.\n"
    "Answer: E.\n"
    "Question: Where do most short-period comets come from and how do we know?\n"
    "A) The Kuiper belt; short period comets tend to be in the plane of the solar system like the Kuiper belt.\n"
    "B) The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.\n"
    "C) The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.\n"
    "D) The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.\n"
    "E) The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.\n"
    "F) The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.\n"
    "G) The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\n"
    "Answer: Let's think step by step. Most short-period comets originate from the Kuiper belt. This is deduced from the observation that these comets tend to follow orbits that lie in the plane of the solar system, similar to the distribution of objects in the Kuiper belt itself. Thus, the alignment of these cometary orbits with the ecliptic plane points to their Kuiper belt origin.\n"
    "Answer: A.\n"
    "Question: Colors in a soap bubble result from light ( ).\n"
    "A) dispersion\nB) deflection\nC) refraction\nD) reflection\nE) interference\nF) converted to a different frequency\nG) polarization\nH) absorption\nI) diffraction\nJ) transmission\n"
    "Answer: Let's think step by step.The colorful patterns observed in a soap bubble are caused by the phenomenon of light interference. This occurs when light waves bounce between the two surfaces of the soap film, combining constructively or destructively based on their phase differences and the varying thickness of the film. These interactions result in vibrant color patterns due to variations in the intensity of different wavelengths of light.\n"
    "Answer: E.\n"
    "Question: A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?\n"
    "A) 240 W\nB) 120 W\nC) 10 W\nD) 480 W\nE) 360 W\nF) 200 W\nG) 30 W\nH) 150 W\nI) 60 W\nJ) 300 W\n"
    "Answer: Let's think step by step. The rate of energy usage, known as power, in an electrical circuit is calculated by the product of voltage and current. For a microwave oven connected to a 120 V outlet and drawing a current of 2 amps, the power consumption can be calculated as follows:\n"
    "Power = Voltage × Current = 120 V × 2 A = 240 W.\n"
    "Therefore, the microwave oven uses energy at a rate of 240 watts.\n"
    "Answer: A.\n"
)


def super_gpqa_multishot_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    q = doc["question"]
    choices = _get_choices(doc)
    current_lines = [f"Question: {q}", ""]
    for letter in sorted(choices.keys()):
        current_lines.append(f"{letter}) {choices[letter]}")
    current_block = "\n".join(current_lines)
    return FEWSHOT_PROMPT + "\n" + current_block + "\nAnswer: Let's think step by step."
