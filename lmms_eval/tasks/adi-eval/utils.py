from io import BytesIO

import pint
import skimage.io as skio
from PIL import Image
from pint import UnitRegistry, errors
from word2num import word2num

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
TOLERANCE_SCALE = 4


def adi_eval_doc_to_visual(doc):
    with BytesIO(doc["image"]) as buf:
        img = skio.imread(buf)
        img = Image.fromarray(img)
    return [img.convert("RGB")]


def adi_eval_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def adi_eval_process_results(doc, results):
    pred = results[0]
    type = doc["dataset"]
    score, reason = tolerance_correctness(pred, doc["answer"], doc["axis_scale"])
    score = 1.0 if score else 0.0
    return_dict = {"overall": score}
    return_dict[doc["type"]] = score
    return_dict["reason"] = reason
    if type == "human":
        return_dict["human"] = score
    else:
        return_dict["augmented"] = score
    return return_dict


def tolerance_correctness(prediction, target, axis_scale) -> bool:
    if prediction == target:
        return True, "Exact_Match"

    simplified_prediction = prediction.replace("µ", "u").lower().rstrip(".")
    simplified_target = target.replace("µ", "u").lower().rstrip(".")
    if simplified_prediction == simplified_target:
        return True, "Match_on_Simplified_String"
    if simplified_prediction.split("=")[-1].strip() == simplified_target:
        return True, "Match_with_Equals"

    if axis_scale is not None:
        try:
            prediction_unit = ureg(prediction.replace("v", "V"))
        except:
            pass
        else:
            target_unit = ureg(target.replace("v", "V"))
            if type(target_unit) == type(prediction_unit):
                if isinstance(prediction_unit, pint.Quantity):
                    try:
                        prediction_unit = prediction_unit.to(target_unit.units)
                    except errors.DimensionalityError:
                        return False, "Unconvertable_Units"
                    delta = abs((prediction_unit - target_unit).magnitude)
                else:
                    delta = abs(prediction_unit - target_unit)
                tolerance = axis_scale / TOLERANCE_SCALE
                if delta <= tolerance:
                    return True, "Match_In_Tolerance"
                else:
                    return False, "Not_Matched_Outside_Tolerance"
            else:
                return False, "Missing/Spurious_Units"

    numberified_prediction = prediction
    try:
        converted = word2num(prediction)
        numberified_prediction = converted if converted is not None else prediction
    except:
        pass
    try:
        if float(numberified_prediction) == float(target):
            return True, "Match_on_numberified_prediction"
    except ValueError as e:
        pass

    return False, "Unmatched_String_Compare"
