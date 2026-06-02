import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def sat_doc_to_visual(doc):
    images = doc["image_bytes"]
    images = [each.convert("RGB") for each in images]
    return images


def sat_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = (
        doc["question"].strip()
        + "\n"
        + "When multiple images are provided, You want to realize that when camera moves left, the objects in the image move right, and vice versa. And when you are asked about the movement of the same object captured in different images, you can compare the position of the object relative to other objects in the image to answer the question.\n"
    )
    candidates = doc["answers"]
    candidates_str = ""
    for i, candidate in enumerate(candidates):
        option_letter = chr(ord("A") + i)
        candidates_str += f"{option_letter}. {candidate}\n"

    return question + "\n" + candidates_str + "\n" + "Only output the letter of the correct option and nothing else."


def sat_doc_to_target(doc):
    return chr(ord("A") + doc["correct_answer_idx"])


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0].split(" ")[0])

        return filtered_resps
