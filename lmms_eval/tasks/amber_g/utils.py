import os
import nltk
import json
import spacy
import warnings
import numpy as np
from nltk.stem import WordNetLemmatizer
from PIL import Image
from pathlib import Path
import datasets

# Configuration
AMBER_BASE_DIR = os.environ.get("AMBER_BASE_DIR", "./data/amber")
SIMILARITY_THRESHOLD = 0.8
EVALUATION_TYPE = 'g'  # Default to g
METADATA_DIR = os.path.join(AMBER_BASE_DIR, "metadata")
QUESTIONS_DIR = os.path.join(AMBER_BASE_DIR, "questions")

# Global variables for loaded metadata
_ASSOCIATION = None
_HALLUCINATION_WORDS = None
_SAFE_WORDS = None
_ANNOTATIONS = None
_METRICS_INIT = None
_NLP = None  # Lazy load spaCy

def get_nlp():
    """Lazy load spaCy model."""
    global _NLP
    if _NLP is not None:
        return _NLP
    
    models_to_try = ["en_core_web_md", "en_core_web_sm", "en_core_web_lg"]
    
    for model_name in models_to_try:
        try:
            _NLP = spacy.load(model_name)
            return _NLP
        except OSError:
            continue
    
    raise OSError(
        "No spaCy model found. Please install one:\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm\n"
        "or for better accuracy:\n"
        "  python -m spacy download en_core_web_md"
    )



def load_metadata():
    """Load AMBER metadata files once."""
    global _ASSOCIATION, _HALLUCINATION_WORDS, _SAFE_WORDS, _ANNOTATIONS, _METRICS_INIT
    
    if _ASSOCIATION is not None:
        return
    
    association_file = os.path.join(METADATA_DIR, 'relation.json')
    _ASSOCIATION = load_json(association_file)
    _HALLUCINATION_WORDS = set()
    for word1, related in _ASSOCIATION.items():
        _HALLUCINATION_WORDS.add(word1)
        _HALLUCINATION_WORDS.update(related)
    
    # Load safe words
    safe_words_file = os.path.join(METADATA_DIR, 'safe_words.txt')
    _SAFE_WORDS = load_text_lines(safe_words_file)
    
    # Load annotations
    annotation_file = os.path.join(METADATA_DIR, 'annotations.json')
    _ANNOTATIONS = load_json(annotation_file)
    
    # Load metrics initialization
    metrics_file = os.path.join(METADATA_DIR, 'metrics.txt')
    _METRICS_INIT = load_metrics(metrics_file)

#########################################
#          Utility Functions           #
########################################

def load_json(file_path):
    """Load JSON file and return the data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_text_lines(file_path):
    """Load text file and return list of stripped lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def load_metrics(metrics_path):
    """Initialize and return a metrics dict based on a metrics file with key=value lines."""
    metrics = {}
    with open(metrics_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split('=')
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value
    return metrics


def check_synonyms_word(word1, word2, similarity_threshold):
    """Check if two words are synonyms based on spaCy similarity."""
    nlp = get_nlp()  # Lazy load spaCy model
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_threshold


def extract_nouns(text):
    """Extract lemmatized nouns from given text using NLTK."""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns


########################################
#       Evaluation Computations        #
########################################

def setup_dimensions(evaluation_type):
    """Setup which evaluation dimensions to run based on the evaluation_type argument."""
    dimensions = {'g': False, 'de': False, 'da': False, 'dr': False}
    if evaluation_type == 'a':
        for key in dimensions:
            dimensions[key] = True
    elif evaluation_type == 'g':
        dimensions['g'] = True
    elif evaluation_type == 'd':
        dimensions['de'] = True
        dimensions['da'] = True
        dimensions['dr'] = True
    else:
        dimensions[evaluation_type] = True
    return dimensions


def prepare_association(association):
    """Load word associations and return a set of hallucination words."""
    hallucination_words = set()
    for word1, related in association.items():
        hallucination_words.add(word1)
        hallucination_words.update(related)
    return association, hallucination_words


def process_generative_task(data_item, ground_truth_item, association, hallucination_words,
                            global_safe_words, similarity_threshold, metrics):
    """Process a generative task item and update the metrics dictionary accordingly."""
    question_id = data_item['question_id']
    nouns = extract_nouns(data_item['text'])
    filtered_nouns = [noun for noun in nouns if noun in hallucination_words]

    safe_words = []
    safe_list = []
    for idx, word in enumerate(ground_truth_item['truth']):
        related = association.get(word, [])
        safe_words += related
        safe_list += [idx] * len(related)

    ha_words = []
    ha_list = []
    for idx, word in enumerate(ground_truth_item['hallu']):
        related = association.get(word, [])
        ha_words += related
        ha_list += [idx] * len(related)

    safe_words += ground_truth_item['truth']
    safe_len = len(ground_truth_item['truth'])
    safe_list += [0] * safe_len

    ha_words += ground_truth_item['hallu']
    ha_len = len(ground_truth_item['hallu'])
    ha_list += [0] * ha_len

    safe_flag_list = [0] * len(filtered_nouns)

    for idx, noun in enumerate(filtered_nouns):
        if noun in global_safe_words:
            continue

        if noun in safe_words:
            for j in range(len(safe_words)):
                if noun == safe_words[j]:
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                    break
            continue

        if noun in ha_words:
            for j in range(len(ha_words)):
                if noun == ha_words[j]:
                    if j < (len(ha_list) - ha_len):
                        ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                    else:
                        ha_list[j] = 1
                    break

        for j, check_word in enumerate(ha_words):
            if check_synonyms_word(noun, check_word, similarity_threshold):
                if j < (len(ha_list) - ha_len):
                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                else:
                    ha_list[j] = 1
                break

        flag = False
        for j, check_word in enumerate(safe_words):
            if check_synonyms_word(noun, check_word, similarity_threshold):
                flag = True
                if j < (len(safe_list) - safe_len):
                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                else:
                    safe_list[j] = 1
                break
        if flag:
            continue

        safe_flag_list[idx] = 1

    metrics['chair_score'] += sum(safe_flag_list)
    metrics['chair_num'] += len(safe_flag_list)
    metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
    metrics['safe_cover_num'] += len(safe_list[-safe_len:])
    metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
    metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
    if sum(safe_flag_list) == 0:
        metrics['non_hallu_score'] += 1
    metrics['non_hallu_num'] += 1


def process_discriminative_task(data_item, ground_truth_item, metrics):
    """Process a discriminative task item and update the metrics dictionary accordingly."""
    question_id = data_item['question_id']
    metrics['qa_correct_num'] += 1

    gt_type = ground_truth_item['type']
    if gt_type == 'discriminative-attribute-state':
        metrics['as_qa_correct_num'] += 1
    elif gt_type == 'discriminative-attribute-number':
        metrics['an_qa_correct_num'] += 1
    elif gt_type == 'discriminative-attribute-action':
        metrics['aa_qa_correct_num'] += 1
    elif gt_type == 'discriminative-hallucination':
        metrics['ha_qa_correct_num'] += 1
    else:
        metrics['asso_qa_correct_num'] += 1

    truth = ground_truth_item['truth']
    response = data_item['text']
    if "yes" in response.lower():
        response = "Yes"
    elif "no" in response.lower():
        response = "No"

    if truth == 'yes':
        if response == "Yes":
            metrics['qa_correct_score'] += 1
            if gt_type == 'discriminative-attribute-state':
                metrics['as_qa_correct_score'] += 1
            elif gt_type == 'discriminative-attribute-number':
                metrics['an_qa_correct_score'] += 1
            elif gt_type == 'discriminative-attribute-action':
                metrics['aa_qa_correct_score'] += 1
            elif gt_type == 'discriminative-hallucination':
                metrics['ha_qa_correct_score'] += 1
            else:
                metrics['asso_qa_correct_score'] += 1
    else:
        metrics['qa_no_num'] += 1
        if gt_type == 'discriminative-attribute-state':
            metrics['as_qa_no_num'] += 1
        elif gt_type == 'discriminative-attribute-number':
            metrics['an_qa_no_num'] += 1
        elif gt_type == 'discriminative-attribute-action':
            metrics['aa_qa_no_num'] += 1
        elif gt_type == 'discriminative-hallucination':
            metrics['ha_qa_no_num'] += 1
        else:
            metrics['asso_qa_no_num'] += 1

        if response == "No":
            metrics['qa_correct_score'] += 1
            metrics['qa_no_score'] += 1
            if gt_type == 'discriminative-attribute-state':
                metrics['as_qa_correct_score'] += 1
                metrics['as_qa_no_score'] += 1
            elif gt_type == 'discriminative-attribute-number':
                metrics['an_qa_correct_score'] += 1
                metrics['an_qa_no_score'] += 1
            elif gt_type == 'discriminative-attribute-action':
                metrics['aa_qa_correct_score'] += 1
                metrics['aa_qa_no_score'] += 1
            elif gt_type == 'discriminative-hallucination':
                metrics['ha_qa_correct_score'] += 1
                metrics['ha_qa_no_score'] += 1
            else:
                metrics['asso_qa_correct_score'] += 1
                metrics['asso_qa_no_score'] += 1

    if response == "No":
        metrics['qa_ans_no_num'] += 1
        if gt_type == 'discriminative-attribute-state':
            metrics['as_qa_ans_no_num'] += 1
        elif gt_type == 'discriminative-attribute-number':
            metrics['an_qa_ans_no_num'] += 1
        elif gt_type == 'discriminative-attribute-action':
            metrics['aa_qa_ans_no_num'] += 1
        elif gt_type == 'discriminative-hallucination':
            metrics['ha_qa_ans_no_num'] += 1
        else:
            metrics['asso_qa_ans_no_num'] += 1
        if truth == 'no':
            metrics['qa_ans_no_score'] += 1
            if gt_type == 'discriminative-attribute-state':
                metrics['as_qa_ans_no_score'] += 1
            elif gt_type == 'discriminative-attribute-number':
                metrics['an_qa_ans_no_score'] += 1
            elif gt_type == 'discriminative-attribute-action':
                metrics['aa_qa_ans_no_score'] += 1
            elif gt_type == 'discriminative-hallucination':
                metrics['ha_qa_ans_no_score'] += 1
            else:
                metrics['asso_qa_ans_no_score'] += 1


########################################
#        lmms-eval Interface           #
########################################

def amber_g_doc_to_visual(doc):
    """Convert document to visual input."""
    # Read num_image from environment variable
    num_image = int(os.environ.get("NUM_IMAGE", "1"))
    
    if num_image == 1:
        # print("one image!")
        return [doc["image"].convert("RGB")]
    elif num_image == 2:
        # print("two images!")
        # Use the same image twice (similar to mmbench pattern)
        return [doc["image"].convert("RGB"), doc["image"].convert("RGB")]
    else:
        raise ValueError(f"num_image must be 1 or 2, got {num_image}")


def amber_g_doc_to_text(doc):
    """Convert document to text prompt."""
    task_type = doc.get("task_type", "generative")
    
    if task_type == 'generative':
        return "Describe this image"
    else:
        return doc.get("text", "")


def amber_g_process_result(doc, result):
    """Process a single result for AMBER evaluation."""
    load_metadata()
    pred_text = result[0] if len(result) > 0 else ""
    question_id = doc.get("question_id", 0)
    
    gt_item = {
        'question_id': question_id,
        'type': doc.get('task_type', 'generative'),
        'truth': doc.get('truth', []),
        'hallu': doc.get('hallu', [])
    }

    data_item = {
        'question_id': question_id,
        'text': pred_text
    }
    
    # Initialize temporary metrics
    temp_metrics = _METRICS_INIT.copy()
    
    if gt_item['type'] == 'generative':
        process_generative_task(data_item, gt_item, _ASSOCIATION, _HALLUCINATION_WORDS,
                                _SAFE_WORDS, SIMILARITY_THRESHOLD, temp_metrics)
        
        return {
            'amber_chair': temp_metrics.copy(),
            'amber_cover': temp_metrics.copy(),
            'amber_hal': temp_metrics.copy(),
            'amber_cog': temp_metrics.copy(),
        }
    else:
        process_discriminative_task(data_item, gt_item, temp_metrics)
        
        return {
            'amber_chair': temp_metrics,
            'amber_cover': temp_metrics,
            'amber_hal': temp_metrics,
            'amber_cog': temp_metrics,
        }


def amber_g_aggregate_chair(results):
    """Aggregate CHAIR metric."""
    chair_score = sum(r['chair_score'] for r in results if r is not None)
    chair_num = sum(r['chair_num'] for r in results if r is not None)
    
    if chair_num == 0:
        return 0.0
    return (chair_score / chair_num) * 100


def amber_g_aggregate_cover(results):
    """Aggregate Cover metric."""
    safe_cover_score = sum(r['safe_cover_score'] for r in results if r is not None)
    safe_cover_num = sum(r['safe_cover_num'] for r in results if r is not None)
    
    if safe_cover_num == 0:
        return 0.0
    return (safe_cover_score / safe_cover_num) * 100


def amber_g_aggregate_hal(results):
    """Aggregate Hal metric."""
    non_hallu_score = sum(r['non_hallu_score'] for r in results if r is not None)
    non_hallu_num = sum(r['non_hallu_num'] for r in results if r is not None)
    
    if non_hallu_num == 0:
        return 0.0
    return 100 - (non_hallu_score / non_hallu_num) * 100


def amber_g_aggregate_cog(results):
    """Aggregate Cog metric."""
    hallu_cover_score = sum(r['hallu_cover_score'] for r in results if r is not None)
    hallu_cover_num = sum(r['hallu_cover_num'] for r in results if r is not None)
    
    if hallu_cover_num == 0:
        return 0.0
    return (hallu_cover_score / hallu_cover_num) * 100
