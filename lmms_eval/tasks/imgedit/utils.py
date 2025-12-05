"""
ImgEdit Benchmark Utils
Image editing evaluation task using GPT-4o or Qwen2.5-VL

Based on: https://github.com/sysuyy/ImgEdit
Paper: ImgEdit: A Unified Image Editing Benchmark

Environment variables:
    - IMGEDIT_EVAL_BACKBONE: "gpt4o" or "qwen25vl" (default: "gpt4o")
    - IMGEDIT_MODEL_NAME: Name of the model being evaluated
    - IMGEDIT_OUTPUT_DIR: Directory to save generated images
    - IMGEDIT_ORIGIN_IMG_ROOT: Root directory of original images
    - OPENAI_API_KEY: OpenAI API key (for GPT-4o)
    - OPENAI_BASE_URL: Optional custom OpenAI API base URL
    - QWEN_MODEL_PATH: Path to Qwen2.5-VL model (default: Qwen/Qwen2.5-VL-72B-Instruct-AWQ)
"""

import base64
import json
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

# Global variable to cache Qwen2.5-VL model (lazy loading)
_qwen25vl_model = None

# Edit type prompts for evaluation
# These are the evaluation criteria for different edit types
IMGEDIT_PROMPTS = {
    "replace": """
You are a data rater specializing in grading image replacement edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the replacement editing effect on a 5-point scale from three perspectives:

Prompt Compliance
1  Target not replaced, or an unrelated object edited.
2  Only part of the target replaced, or wrong class/description used.
3  Target largely replaced but other objects altered, remnants visible, or count/position clearly wrong.
4  Correct object fully replaced; only minor attribute errors (colour, size, etc.).
5  Perfect replacement: all and only the specified objects removed; new objects' class, number, position, scale, pose and detail exactly match the prompt.

Visual Naturalness
1  Image heavily broken or new object deformed / extremely blurred.
2  Obvious seams, smears, or strong mismatch in resolution or colour; background not restored.
3  Basic style similar, but lighting or palette clashes; fuzzy edges or noise are noticeable.
4  Style almost uniform; tiny edge artefacts visible only on close inspection; casual viewers see no edit.
5  Completely seamless; new objects blend fully with the scene, edit area undetectable.

Physical & Detail Integrity
1  Floating, interpenetration, severe perspective/light errors; key original elements ruined; background heavily warped.
2  Missing shadows/occlusion; large background shifts or holes.
3  Lighting, perspective and contact surfaces mostly correct; small but tolerable errors; background adjusted locally.
4  New objects interact realistically with scene (shadows, reflections, texture) and preserve existing details; background change minimal.
5  Physically flawless and enhances realism: accurate highlights, shadows, reflections, ambient effects; background untouched.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical & Detail Integrity: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "add": """
You are a data rater specializing in grading image addition edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the added object(s) on a 5-point scale from three perspectives:

Prompt Compliance
1  Nothing added or the added content is corrupt.
2  Added object is a wrong class or unrelated to the prompt.
3  Correct class, but key attributes (position, colour, size, count, etc.) are wrong.
4  Main attributes correct; only minor details off or 1-2 small features missing.
5  Every stated attribute correct and scene logic reasonable; only microscopic flaws.

Visual Naturalness
1  Image badly broken or full of artefacts.
2  Obvious paste marks; style, resolution, or palette strongly mismatch.
3  General style similar, but lighting or colours clearly clash; noticeable disharmony.
4  Style almost uniform; small edge issues visible only when zoomed.
5  Perfect blend; no visible difference between added object and original image.

Physical & Detail Coherence
1  Severe physical errors (floating, wrong perspective/light); key original elements blocked; background heavily distorted.
2  Contact or occlusion handled poorly; minor background shifts, jaggies or noise; background visibly changed.
3  Lighting, perspective, and contact mostly correct; remaining flaws small and acceptable; limited background change.
4  Shadows, reflections, and material response believable; no loss of original detail; background changes are minute.
5  Added object enhances overall realism: precise highlights, shadows, ambient effects; background essentially untouched.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical & Detail Coherence: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "adjust": """
You are a data rater specializing in grading attribute alteration edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the attribute change on a 5-point scale from three perspectives:

Prompt Compliance
1  Target not adjusted, wrong object touched, or geometry changed.
2  Right object but wrong attribute value/direction; only part edited; other objects also altered; slight stretch/crop.
3  Mainly correct object and attribute, yet large hue/brightness/texture error; minor collateral edits; visible jaggies/distortion.
4  All requested objects adjusted, only their attributes changed; shape kept; small inaccuracy in colour, material or amount.
5  Exactly and only the requested objects adjusted; colour, material, gloss etc. match the prompt perfectly; shape 100% intact; zero unintended edits.

Visual Seamlessness
1  Massive colour spill, mosaics or heavy noise; image nearly unusable.
2  Clear smears/bleeding on edges; abrupt resolution or tone shift; highlights/shadows clipped; background gaps.
3  Overall palette OK but local tone or grain conflicts; soft edges; noticeable disharmony.
4  Style unified, transitions smooth; only slight edge artefacts visible when zoomed.
5  No detectable edit traces; colours/materials fuse with scene lighting; edit area practically invisible.

Physical & Detail Fidelity
1  Object floating, interpenetrating, or severe perspective/light mismatch; background badly warped.
2  Missing shadows/highlights; wrong reflection direction; background visibly discoloured or distorted.
3  Light, perspective and contact surface largely correct; minor acceptable flaws; background only locally affected.
4  Adjusted material interacts believably with scene; shadows, highlights, reflections handled well; original details preserved.
5  High physical realism: fine micro-highlights, diffuse bounce, subsurface effects present; overall scene realism improved.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Seamlessness: A number from 1 to 5.
Physical & Detail Fidelity: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "remove": """
You are a data rater specializing in grading object removal edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the removal quality on a 5-point scale from three perspectives:

Prompt Compliance
1  Nothing removed, or an unrelated object edited.
2  Target only partly removed, or a different instance/class deleted, or another object appears in the gap.
3  Target mostly removed but extra objects also deleted, or fragments of the target remain.
4  Only the specified objects removed, but a few tiny/background items deleted by mistake, or the count is wrong.
5  Perfect: all and only the requested objects removed; every other element untouched.

Visual Naturalness
1  Image badly broken (large holes, strong artefacts).
2  Clear erase marks; colour/resolution mismatch; background not restored.
3  General look acceptable yet lighting/colour/style still clash; blur or noise visible.
4  Style consistent; minor edge issues visible only when zoomed.
5  Seamless: removal is virtually impossible to spot.

Physical & Detail Integrity
1  Severe physical errors (floating items, wrong perspective/light); key scene elements damaged; background heavily warped.
2  Large un-filled gaps or obvious background shifts.
3  Lighting, perspective and contacts mostly correct; flaws small and tolerable; background adjusted locally.
4  Background reconstruction clean; existing details preserved; only minute changes outside the removal area.
5  Physically flawless and even enhances realism: accurate light/shadow/texture infill, high-quality micro-details.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical & Detail Integrity: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "style": """
You are a data rater specializing in grading style transfer edits. You will be given an input image, a reference style, and the styled result. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:

Style Fidelity
1  Target style absent or clearly wrong.
2  Style shows in a few areas only, or mixed with unrelated styles.
3  Key traits (palette, brushwork, texture) present but patchy or inconsistent.
4  Style reproduced across almost the whole image; only small local mismatches.
5  Full, faithful transfer: colour, texture, brushwork, lighting all match the exemplar over the entire image.

Content Preservation
1  Major objects or layout lost/distorted; original scene barely recognisable.
2  Main subject recognisable, but size, perspective or key parts clearly wrong/missing.
3  Overall structure correct; some local warping or minor omissions.
4  Nearly all geometry intact; only slight, non-distracting deformation.
5  All objects and spatial relations kept; only stylistic, harmless distortion.

Rendering Quality
1  Heavy noise, banding, pixel damage or blur; image unusable.
2  Visible seams, aliasing, colour drift; low resolution or chaotic strokes.
3  Moderate quality: local blur/noise/texture breaks, but generally acceptable.
4  Sharp, coherent strokes; tiny artefacts visible only when zoomed.
5  High resolution, no artefacts; strokes, textures and colour transitions look fully natural.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Style Fidelity: A number from 1 to 5.
Content Preservation: A number from 1 to 5.
Rendering Quality: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the input, reference style, and styled output image:
""",
    "action": """
You are a data rater specializing in grading action or expression change edits. You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the motion or expression change on a 5-point scale from three perspectives:

Action / Expression Fidelity
1  No visible change, or wrong action / expression.
2  Partial or clearly incorrect pose; only some body parts change; expression direction wrong.
3  Main idea present but details off (angle, side, intensity, missing gesture).
4  Requested pose / expression achieved with just minor inaccuracy (small angular drift, timing nuance).
5  Exact match to prompt: every limb, gesture, and facial muscle aligns with the described action.

Identity Preservation
1  Person unrecognisable; face or body replaced.
2  Strong drift: key facial features, hairstyle or clothing heavily altered.
3  Mostly same identity; moderate changes in some features but still recognisable.
4  Identity clearly the same; only subtle stylisation or lighting differences.
5  Perfect preservation of face, hairstyle, skin tone, clothing and accessories.

Visual & Anatomical Coherence
1  Severe artifacts: broken or duplicated limbs, extreme distortion, heavy noise/blur.
2  Noticeable cut-out halos, proportion errors, lighting or perspective clearly off.
3  Generally plausible; minor joint or shading issues; small noise/blur acceptable.
4  Clean render; anatomy, lighting, depth and edges consistent; flaws only on close inspection.
5  Flawless realism or stylistic coherence; perfect anatomy, lighting, shadows and texture continuity.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Action Fidelity: A number from 1 to 5.
Identity Preservation: A number from 1 to 5.
Visual & Anatomical Coherence: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "extract": """
You are a data rater specializing in grading object cut-out quality. You will be given an image with the object extracted on a white background. Your task is to evaluate the cut-out accuracy on a 5-point scale from three perspectives:

Object Selection & Identity
1  Wrong object or multiple objects extracted.
2  Correct class but only part of the object, or obvious intrusions from other items.
3  Object largely correct yet small pieces missing / extra, identity still recognisable.
4  Full object with clear identity; only tiny mis-crop (e.g., tip of antenna).
5  Exact requested object, complete and unmistakably the same instance (ID).

Mask Precision & Background Purity
1  Large background remnants, holes in mask, or non-white backdrop dominates.
2  Noticeable jagged edges, colour fringes, grey/colour patches in white area.
3  Acceptable mask; minor edge softness or faint halo visible on close look.
4  Clean, smooth edges; white (#FFFFFF) background uniform, tiny artefacts only when zoomed.
5  Crisp anti-aliased contour, zero spill or halo; backdrop perfectly pure white throughout.

Object Integrity & Visual Quality
1  Severe blur, compression, deformation, or missing parts; unusable.
2  Moderate noise, colour shift, or slight warping; details clearly degraded.
3  Overall intact with minor softness or noise; colours mostly preserved.
4  Sharp detail, accurate colours; negligible artefacts.
5  Pristine: high-resolution detail, true colours, no artefacts or distortion.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Object Identity: A number from 1 to 5.
Mask Precision: A number from 1 to 5.
Visual Quality: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below is the extracted object image:
""",
    "background": """
You are a data rater specializing in grading background editing. You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:

Instruction Compliance
1  No change, or background unrelated to prompt, or foreground also replaced/distorted.
2  Background partly replaced or wrong style/content; foreground noticeably altered.
3  Main background replaced but elements missing/extra, or faint spill onto subject edges.
4  Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).
5  Background exactly matches prompt (content, style, placement); all foreground pixels untouched.

Visual Seamlessness (Edge & Texture Blend)
1  Large tearing, posterisation, extreme blur/noise; edit area obvious at a glance.
2  Clear cut-out halos, colour-resolution gap, or heavy smudge strokes.
3  Blend acceptable but visible on closer look: slight edge blur, grain or palette shift.
4  Nearly invisible seams; textures and sharpness aligned, only minor issues when zoomed in.
5  Indistinguishable composite: edges, textures, resolution and colour grading perfectly continuous.

Physical Consistency (Lighting, Perspective, Depth)
1  Severe mismatch: wrong horizon, conflicting light direction, floating subject, warped geometry.
2  Noticeable but not extreme inconsistencies in light, shadows or scale; depth cues off.
3  Overall believable; small errors in shadow length, perspective or ambient colour.
4  Lighting, scale, depth, and camera angle well matched; only subtle discrepancies.
5  Physically flawless: foreground and new background share coherent light, shadows, reflections, perspective and atmospheric depth, enhancing overall realism.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Instruction Compliance: A number from 1 to 5.
Visual Seamlessness: A number from 1 to 5.
Physical Consistency: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
    "compose": """
You are a data rater specializing in grading hybrid image edits (involving multiple operations on multiple objects). You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the overall editing quality on a 5-point scale from three perspectives:

Instruction Compliance
1  Neither object nor operations match the prompt; wrong items edited or shapes distorted.
2  Only one object correctly edited, or both edited but with wrong/partial operations; collateral changes to other items.
3  Both target objects touched, each with the requested operation broadly correct but missing details (e.g., wrong colour value, incomplete removal).
4  Both objects receive the exact operations; tiny deviations in amount, position, or parameter. No unintended edits elsewhere.
5  Perfect execution: each object fully reflects its specified operation, all other scene elements untouched.

Visual Naturalness (Seamlessness)
1  Large artefacts, obvious cut-outs, heavy blur/noise; edits conspicuous at a glance.
2  Clear edge halos, colour or resolution mismatch, awkward scaling.
3  Acceptable but visible on close look: slight edge softness, minor palette or focus shift.
4  Edits blend smoothly; seams hard to spot, textures and sharpness largely consistent.
5  Indistinguishable composite: colour grading, grain, resolution and style fully match the original image.

Physical Consistency & Fine Detail
1  Severe lighting/perspective mismatch, missing or wrong shadows; objects appear floating or warped.
2  Noticeable but tolerable inconsistencies in illumination, scale, or depth cues.
3  Generally plausible; small errors in shadow length, reflection angle, or texture alignment.
4  Lighting, perspective, and material response closely match; only subtle flaws visible when zoomed.
5  Physically flawless: shadows, highlights, reflections, depth and texture perfectly integrated, enhancing overall realism.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Instruction Compliance: A number from 1 to 5.
Visual Naturalness: A number from 1 to 5.
Physical Consistency & Fine Detail: A number from 1 to 5.
editing instruction is : <edit_prompt>.

Below are the images before and after editing:
""",
}

# Edit types supported
IMGEDIT_EDIT_TYPES = [
    "replace",
    "add",
    "adjust",
    "remove",
    "style",
    "action",
    "extract",
    "background",
    "compose",
]


def image_to_base64(image) -> Optional[str]:
    """Convert PIL Image or image path to base64 string"""
    try:
        if isinstance(image, str):
            # It's a path
            if not os.path.exists(image):
                eval_logger.warning(f"Image file not found: {image}")
                return None
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif hasattr(image, "save"):
            # It's a PIL Image
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            eval_logger.warning(f"Unknown image type: {type(image)}")
            return None
    except Exception as e:
        eval_logger.error(f"Error converting image to base64: {e}")
        return None


def parse_gpt_scores(response_text: str) -> Tuple[float, float, float]:
    """
    Parse GPT/Qwen response to extract three scores.
    Returns tuple of (score1, score2, score3)
    """
    try:
        # Find all numbers in the format "Score Name: X"
        score_pattern = r":\s*(\d+)"
        matches = re.findall(score_pattern, response_text)

        if len(matches) >= 3:
            # Take the last 3 numbers (the actual scores)
            scores = [float(matches[-3]), float(matches[-2]), float(matches[-1])]
            return tuple(scores)

        # Alternative: find standalone numbers on lines
        lines = response_text.strip().split("\n")
        scores = []
        for line in lines:
            # Look for patterns like "Prompt Compliance: 4" or just "4"
            match = re.search(r"(\d+)\s*$", line.strip())
            if match:
                scores.append(float(match.group(1)))

        if len(scores) >= 3:
            return (scores[-3], scores[-2], scores[-1])

        eval_logger.warning(f"Could not parse 3 scores from response: {response_text[:200]}...")
        return (0.0, 0.0, 0.0)
    except Exception as e:
        eval_logger.error(f"Error parsing scores: {e}")
        return (0.0, 0.0, 0.0)


def calculate_average_score(scores: Tuple[float, float, float]) -> float:
    """Calculate average of three scores"""
    return sum(scores) / 3.0


def imgedit_doc_to_visual(doc):
    """
    Extract input image from document.

    Priority order:
    1. input_image field (PIL Image from embedded dataset)
    2. image_path field (full path saved by prepare_dataset.py)
    3. id field + IMGEDIT_ORIGIN_IMG_ROOT (relative path like "animal/000342021.jpg")
    4. input_image as string path
    """
    origin_img_root = os.getenv("IMGEDIT_ORIGIN_IMG_ROOT", "")

    # 1. Try input_image field (PIL Image from embedded dataset)
    input_image = doc.get("input_image") or doc.get("image")
    if input_image is not None and hasattr(input_image, "convert"):
        try:
            return [input_image.convert("RGB")]
        except Exception as e:
            eval_logger.error(f"Error converting input_image: {e}")

    # 2. Try image_path field (full path saved by prepare_dataset.py)
    image_path = doc.get("image_path", "")
    if image_path and os.path.exists(image_path):
        try:
            return [Image.open(image_path).convert("RGB")]
        except Exception as e:
            eval_logger.error(f"Error loading image from image_path {image_path}: {e}")

    # 3. Try id field + origin_img_root (relative path like "animal/000342021.jpg")
    image_id = doc.get("id", "")
    if image_id and origin_img_root:
        full_path = os.path.join(origin_img_root, image_id)
        if os.path.exists(full_path):
            try:
                return [Image.open(full_path).convert("RGB")]
            except Exception as e:
                eval_logger.error(f"Error loading image from {full_path}: {e}")

    # 4. Try input_image as string path
    if input_image is not None and isinstance(input_image, str):
        # Try as absolute path first
        if os.path.exists(input_image):
            try:
                return [Image.open(input_image).convert("RGB")]
            except Exception as e:
                eval_logger.error(f"Error loading image from {input_image}: {e}")
        # Try with origin_img_root
        elif origin_img_root:
            full_path = os.path.join(origin_img_root, input_image)
            if os.path.exists(full_path):
                try:
                    return [Image.open(full_path).convert("RGB")]
                except Exception as e:
                    eval_logger.error(f"Error loading image from {full_path}: {e}")

    eval_logger.warning(f"No input image found in document. " f"Available keys: {list(doc.keys())}, " f"image_path={image_path}, id={image_id}, origin_img_root={origin_img_root}")
    return []


def imgedit_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract instruction text from document"""
    instruction = doc.get("prompt", "").strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def imgedit_doc_to_target(doc):
    """Extract target instruction (for reference)"""
    return doc.get("prompt", "")


# ============================================
# Qwen2.5-VL Evaluation Backend
# ============================================


def _get_qwen25vl_model():
    """
    Get or create Qwen2.5-VL model instance (lazy loading, singleton pattern).
    """
    global _qwen25vl_model
    if _qwen25vl_model is not None:
        return _qwen25vl_model

    try:
        import random

        import numpy as np
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_path = os.getenv("QWEN_MODEL_PATH", "/pfs/training-data/hf/models/Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

        eval_logger.info(f"Loading Qwen2.5-VL model from {model_path}...")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto").eval()
        processor = AutoProcessor.from_pretrained(model_path)

        _qwen25vl_model = {"model": model, "processor": processor, "process_vision_info": process_vision_info}

        eval_logger.info("Qwen2.5-VL model loaded successfully!")
        return _qwen25vl_model

    except ImportError as e:
        eval_logger.error(f"Failed to import Qwen2.5-VL dependencies: {e}")
        eval_logger.error("Please install: pip install transformers qwen-vl-utils")
        return None
    except Exception as e:
        eval_logger.error(f"Failed to load Qwen2.5-VL model: {e}")
        return None


def _call_qwen25vl_for_evaluation(
    original_image,
    edited_image,
    edit_prompt: str,
    edit_type: str,
) -> Optional[str]:
    """
    Call Qwen2.5-VL for image editing evaluation.

    Args:
        original_image: Original image (PIL Image)
        edited_image: Edited image (PIL Image)
        edit_prompt: The editing instruction
        edit_type: Type of edit (replace, add, adjust, etc.)

    Returns:
        Model response text or None if failed
    """
    import random

    import numpy as np
    import torch

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    qwen_model = _get_qwen25vl_model()
    if qwen_model is None:
        return None

    model = qwen_model["model"]
    processor = qwen_model["processor"]
    process_vision_info = qwen_model["process_vision_info"]

    # Get prompt template for this edit type
    prompt_template = IMGEDIT_PROMPTS.get(edit_type, IMGEDIT_PROMPTS["adjust"])
    full_prompt = prompt_template.replace("<edit_prompt>", edit_prompt)

    try:
        # # Build message content for Qwen2.5-VL
        # if edit_type == "extract":
        #     # For extract, only show the edited image
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": edited_image},
        #                 {"type": "text", "text": full_prompt},
        #             ],
        #         }
        #     ]
        # else:
        # For other types, show both original and edited images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": edited_image},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        set_seed(42)

        # Prepare the inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        # Generate output
        generation_config = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": None,
        }
        generated_ids = model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0] if output_text else ""

    except Exception as e:
        eval_logger.error(f"Error calling Qwen2.5-VL: {e}")
        return None


# ============================================
# GPT-4o Evaluation Backend
# ============================================


def _call_gpt_for_evaluation(
    original_image,
    edited_image,
    edit_prompt: str,
    edit_type: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Call GPT-4o for image editing evaluation.

    Args:
        original_image: Original image (PIL Image or path)
        edited_image: Edited image (PIL Image or path)
        edit_prompt: The editing instruction
        edit_type: Type of edit (replace, add, adjust, etc.)
        api_key: OpenAI API key
        base_url: OpenAI API base URL

    Returns:
        GPT response text or None if failed
    """
    try:
        from openai import OpenAI
    except ImportError:
        eval_logger.error("OpenAI package not installed. Run: pip install openai")
        return None

    # Get API credentials from environment if not provided
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        eval_logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return None

    # Convert images to base64
    original_b64 = image_to_base64(original_image)
    edited_b64 = image_to_base64(edited_image)

    if not original_b64 or not edited_b64:
        eval_logger.error("Failed to convert images to base64")
        return None

    # Get prompt template for this edit type
    prompt_template = IMGEDIT_PROMPTS.get(edit_type, IMGEDIT_PROMPTS["adjust"])
    full_prompt = prompt_template.replace("<edit_prompt>", edit_prompt)

    try:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        # Build message content
        content = [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
        ]

        # For extract type, only show the edited image
        if edit_type != "extract":
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}})
        else:
            # For extract, replace the second image with the edited one
            content[1] = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}}

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
        )

        return response.choices[0].message.content
    except Exception as e:
        eval_logger.error(f"Error calling GPT API: {e}")
        return None


# ============================================
# vLLM Qwen Evaluation Backend
# ============================================


def _call_vllm_qwen_for_evaluation(
    original_image,
    edited_image,
    edit_prompt: str,
    edit_type: str,
) -> Optional[str]:
    """
    Call Qwen model via vLLM API for image editing evaluation.

    Environment variables:
        - VLLM_API_BASE: Base URL of vLLM server (e.g., "http://localhost:8000/v1")
        - VLLM_API_KEY: API key if required (default: "EMPTY")
        - VLLM_MODEL_NAME: Model name (default: auto-detect)

    Args:
        original_image: Original image (PIL Image)
        edited_image: Edited image (PIL Image)
        edit_prompt: The editing instruction
        edit_type: Type of edit (replace, add, adjust, etc.)

    Returns:
        Model response text or None if failed
    """
    try:
        from openai import OpenAI
    except ImportError:
        eval_logger.error("OpenAI package not installed. Run: pip install openai")
        return None

    api_base = os.getenv("VLLM_API_BASE")
    if not api_base:
        eval_logger.error("VLLM_API_BASE environment variable not set")
        return None

    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    model_name = os.getenv("VLLM_MODEL_NAME", "default")
    timeout = int(os.getenv("VLLM_TIMEOUT", "120"))

    # Get prompt template for this edit type
    prompt_template = IMGEDIT_PROMPTS.get(edit_type, IMGEDIT_PROMPTS["adjust"])
    full_prompt = prompt_template.replace("<edit_prompt>", edit_prompt)

    # Convert images to base64
    original_b64 = image_to_base64(original_image)
    edited_b64 = image_to_base64(edited_image)

    if not original_b64 or not edited_b64:
        eval_logger.error("Failed to convert images to base64")
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )

        # Auto-detect model name if not set
        if model_name == "default":
            try:
                models = client.models.list()
                if models.data:
                    model_name = models.data[0].id
            except Exception:
                pass

        # Build message content
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}},
            {"type": "text", "text": full_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.1,
        )

        return response.choices[0].message.content if response.choices else ""

    except Exception as e:
        eval_logger.error(f"Error calling vLLM API: {e}")
        return None


# ============================================
# Unified Evaluation Function
# ============================================


def _call_model_for_evaluation(
    original_image,
    edited_image,
    edit_prompt: str,
    edit_type: str,
) -> Optional[str]:
    """
    Call the configured model for evaluation.

    The backend is selected via IMGEDIT_EVAL_BACKBONE environment variable:
    - "gpt4o" (default): Use GPT-4o via OpenAI API
    - "qwen25vl": Use Qwen2.5-VL locally
    - "vllm_qwen" / "vllm_qwen25vl" / "vllm_qwen3vl": Use Qwen via vLLM API
    """
    backbone = os.getenv("IMGEDIT_EVAL_BACKBONE", "gpt4o").lower()

    if backbone == "qwen25vl":
        eval_logger.debug(f"Using Qwen2.5-VL (local) for evaluation (edit_type={edit_type})")
        return _call_qwen25vl_for_evaluation(original_image, edited_image, edit_prompt, edit_type)
    elif backbone in ["vllm_qwen", "vllm_qwen25vl", "vllm_qwen3vl"]:
        eval_logger.debug(f"Using vLLM Qwen for evaluation (edit_type={edit_type})")
        return _call_vllm_qwen_for_evaluation(original_image, edited_image, edit_prompt, edit_type)
    else:
        eval_logger.debug(f"Using GPT-4o for evaluation (edit_type={edit_type})")
        return _call_gpt_for_evaluation(original_image, edited_image, edit_prompt, edit_type)


# ============================================
# Process Results
# ============================================


def imgedit_process_results(doc, results, **kwargs):
    """
    Process model predictions:
    1. Parse JSON output to extract text and images
    2. Save images to required directory structure
    3. Evaluate using GPT-4o or Qwen2.5-VL

    Args:
        doc: Document containing input image, instruction, key, edit_type, etc.
        results: Model predictions [JSON string with {"text": "...", "images": [...]}]
        **kwargs: Additional arguments

    Returns:
        Dict with metrics: imgedit_score1, imgedit_score2, imgedit_score3, imgedit_avg_score
    """
    # Get configuration from environment variables
    model_name = os.getenv("IMGEDIT_MODEL_NAME", "default")
    output_base_dir = os.getenv("IMGEDIT_OUTPUT_DIR", "./logs/imgedit_results")
    origin_img_root = os.getenv("IMGEDIT_ORIGIN_IMG_ROOT", "")

    # Parse prediction
    pred = results[0] if results else "{}"
    try:
        pred = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        eval_logger.warning(f"Failed to parse prediction JSON: {pred}")
        pred = {"text": "", "images": []}

    model_images = pred.get("images", [])

    # Extract document fields
    key = doc.get("key", str(doc.get("id", "unknown")))
    edit_type = doc.get("edit_type", "adjust")
    edit_prompt = doc.get("prompt", "")
    image_id = doc.get("id", "")  # relative path like "animal/000342021.jpg"

    # Get input/original image - try multiple sources in order of priority
    input_image_pil = None

    # 1. Try input_image field (PIL Image from embedded dataset)
    input_image = doc.get("input_image") or doc.get("image")
    if input_image is not None and hasattr(input_image, "convert"):
        try:
            input_image_pil = input_image.convert("RGB")
            eval_logger.debug(f"Loaded input_image from doc (PIL) for key {key}")
        except Exception as e:
            eval_logger.warning(f"Failed to convert input_image for key {key}: {e}")

    # 2. Try image_path field (full path saved by prepare_dataset.py)
    if input_image_pil is None:
        image_path = doc.get("image_path", "")
        if image_path and os.path.exists(image_path):
            try:
                input_image_pil = Image.open(image_path).convert("RGB")
                eval_logger.debug(f"Loaded image from image_path: {image_path}")
            except Exception as e:
                eval_logger.warning(f"Failed to load image from {image_path}: {e}")

    # 3. Try id field with origin_img_root (relative path like "animal/000342021.jpg")
    if input_image_pil is None and image_id and origin_img_root:
        full_path = os.path.join(origin_img_root, image_id)
        if os.path.exists(full_path):
            try:
                input_image_pil = Image.open(full_path).convert("RGB")
                eval_logger.debug(f"Loaded image from origin_img_root + id: {full_path}")
            except Exception as e:
                eval_logger.warning(f"Failed to load image from {full_path}: {e}")

    # 4. Try input_image as string path
    if input_image_pil is None and input_image is not None and isinstance(input_image, str):
        # Try as absolute path first
        if os.path.exists(input_image):
            try:
                input_image_pil = Image.open(input_image).convert("RGB")
                eval_logger.debug(f"Loaded image from input_image path: {input_image}")
            except Exception as e:
                eval_logger.warning(f"Failed to load image from {input_image}: {e}")
        # Try with origin_img_root
        elif origin_img_root:
            full_path = os.path.join(origin_img_root, input_image)
            if os.path.exists(full_path):
                try:
                    input_image_pil = Image.open(full_path).convert("RGB")
                    eval_logger.debug(f"Loaded image from origin_img_root + input_image: {full_path}")
                except Exception as e:
                    eval_logger.warning(f"Failed to load image from {full_path}: {e}")

    # 5. Try to load from saved _SRCIMG file
    if input_image_pil is None:
        src_img_path = os.path.join(output_base_dir, model_name, f"{key}_SRCIMG.png")
        if os.path.exists(src_img_path):
            try:
                input_image_pil = Image.open(src_img_path).convert("RGB")
                eval_logger.debug(f"Loaded source image from _SRCIMG: {src_img_path}")
            except Exception as e:
                eval_logger.warning(f"Failed to load source image from {src_img_path}: {e}")

    # Return zero scores if no input image
    if input_image_pil is None:
        eval_logger.warning(f"No input image found for key {key}. " f"Tried: input_image={doc.get('input_image') is not None}, " f"image_path={doc.get('image_path', '')}, " f"id={image_id}, origin_img_root={origin_img_root}")
        return _create_zero_result(key, edit_type)

    # Find edited image
    edited_image_path = None
    edited_image_pil = None

    if model_images and len(model_images) > 0:
        generated_image_path = model_images[0]
        if os.path.exists(generated_image_path):
            edited_image_path = generated_image_path
            try:
                edited_image_pil = Image.open(edited_image_path).convert("RGB")
            except Exception as e:
                eval_logger.warning(f"Failed to load edited image: {e}")

    # Try to find from standard location
    if edited_image_pil is None:
        existing_path = os.path.join(output_base_dir, model_name, f"{key}.png")
        if os.path.exists(existing_path):
            edited_image_path = existing_path
            try:
                edited_image_pil = Image.open(existing_path).convert("RGB")
            except Exception as e:
                eval_logger.warning(f"Failed to load edited image from {existing_path}: {e}")

    # Return zero scores if no edited image
    if edited_image_pil is None:
        eval_logger.warning(f"No edited image found for key {key}")
        return _create_zero_result(key, edit_type)

    # Call model for evaluation (GPT-4o or Qwen2.5-VL based on config)
    model_response = _call_model_for_evaluation(
        input_image_pil,
        edited_image_pil,
        edit_prompt,
        edit_type,
    )

    if model_response is None:
        eval_logger.warning(f"Model evaluation failed for key {key}")
        return _create_zero_result(key, edit_type)

    # Parse scores from model response
    score1, score2, score3 = parse_gpt_scores(model_response)
    avg_score = calculate_average_score((score1, score2, score3))

    eval_logger.info(f"[{edit_type}] Key {key}: " f"Score1={score1:.1f}, Score2={score2:.1f}, Score3={score3:.1f}, " f"Avg={avg_score:.2f}")

    return {
        "imgedit_score1": {
            "key": key,
            "edit_type": edit_type,
            "score": float(score1),
        },
        "imgedit_score2": {
            "key": key,
            "edit_type": edit_type,
            "score": float(score2),
        },
        "imgedit_score3": {
            "key": key,
            "edit_type": edit_type,
            "score": float(score3),
        },
        "imgedit_avg_score": {
            "key": key,
            "edit_type": edit_type,
            "score": float(avg_score),
            "model_response": model_response,
        },
    }


def _create_zero_result(key: str, edit_type: str) -> Dict:
    """Create a zero-score result dict"""
    return {
        "imgedit_score1": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
        "imgedit_score2": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
        "imgedit_score3": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
        "imgedit_avg_score": {
            "key": key,
            "edit_type": edit_type,
            "score": 0.0,
        },
    }


# ============================================
# Aggregation Functions
# ============================================


def imgedit_aggregate_results(results):
    """
    Aggregate results across all samples and compute final scores.
    Returns overall average score.

    Args:
        results: List of result dicts from process_results

    Returns:
        Final aggregated score (average across all samples)
    """
    if not results:
        return 0.0

    # Calculate average score
    scores = [r["score"] for r in results if "score" in r]
    if not scores:
        return 0.0

    avg_score = np.mean(scores)

    # Log breakdown by edit type
    edit_type_scores = defaultdict(list)

    for r in results:
        if "score" in r:
            edit_type = r.get("edit_type", "unknown")
            edit_type_scores[edit_type].append(r["score"])

    # Log statistics
    eval_logger.info(f"Overall average score: {avg_score:.3f}")
    eval_logger.info(f"Number of samples: {len(scores)}")

    if edit_type_scores:
        eval_logger.info("Scores by edit type:")
        for edit_type, type_scores in sorted(edit_type_scores.items()):
            type_avg = np.mean(type_scores)
            eval_logger.info(f"  {edit_type}: {type_avg:.3f} (n={len(type_scores)})")

    return avg_score


def imgedit_aggregate_by_type(results):
    """
    Aggregate results by edit type and return a dict of scores per type.

    Args:
        results: List of result dicts from process_results

    Returns:
        Dict mapping edit_type to average score
    """
    if not results:
        return {}

    edit_type_scores = defaultdict(list)

    for r in results:
        if "score" in r:
            edit_type = r.get("edit_type", "unknown")
            edit_type_scores[edit_type].append(r["score"])

    type_averages = {}
    for edit_type, type_scores in edit_type_scores.items():
        type_averages[edit_type] = float(np.mean(type_scores))

    # Log the breakdown
    eval_logger.info("=" * 50)
    eval_logger.info("Scores by Edit Type:")
    eval_logger.info("=" * 50)
    for edit_type in IMGEDIT_EDIT_TYPES:
        if edit_type in type_averages:
            eval_logger.info(f"  {edit_type}: {type_averages[edit_type]:.3f} (n={len(edit_type_scores[edit_type])})")
    eval_logger.info("=" * 50)

    return type_averages


# Per-type aggregation functions for YAML
def _aggregate_for_type(results, target_type: str):
    """Helper to aggregate scores for a specific edit type"""
    if not results:
        return 0.0

    type_scores = [r["score"] for r in results if r.get("edit_type") == target_type and "score" in r]

    if not type_scores:
        return 0.0

    return float(np.mean(type_scores))


def imgedit_aggregate_replace(results):
    """Aggregate scores for 'replace' edit type"""
    return _aggregate_for_type(results, "replace")


def imgedit_aggregate_add(results):
    """Aggregate scores for 'add' edit type"""
    return _aggregate_for_type(results, "add")


def imgedit_aggregate_adjust(results):
    """Aggregate scores for 'adjust' edit type"""
    return _aggregate_for_type(results, "adjust")


def imgedit_aggregate_remove(results):
    """Aggregate scores for 'remove' edit type"""
    return _aggregate_for_type(results, "remove")


def imgedit_aggregate_style(results):
    """Aggregate scores for 'style' edit type"""
    return _aggregate_for_type(results, "style")


def imgedit_aggregate_action(results):
    """Aggregate scores for 'action' edit type"""
    return _aggregate_for_type(results, "action")


def imgedit_aggregate_extract(results):
    """Aggregate scores for 'extract' edit type"""
    return _aggregate_for_type(results, "extract")


def imgedit_aggregate_background(results):
    """Aggregate scores for 'background' edit type"""
    return _aggregate_for_type(results, "background")


def imgedit_aggregate_compose(results):
    """Aggregate scores for 'compose' edit type"""
    return _aggregate_for_type(results, "compose")
