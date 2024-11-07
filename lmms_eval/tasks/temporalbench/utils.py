import json
import os
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "temporalbench_short_qa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


textscore_dict, videoscore_dict = {}, {}


def prep_data():
    global textscore_dict, videoscore_dict
    cache_dir = os.path.join(base_cache_dir, cache_name)
    breakpoint()
    with open(os.path.join(cache_dir,  "temporalbench_short_qa.json")) as f:
        textscore_list = json.load(f)
    textscore_dict = {}
    for item in textscore_list:
        textscore_dict[item["idx"]] = item
    return textscore_dict


def temporalbench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir,  doc['video_name'])
    if not os.path.exists(video_path):
        raise Exception(f"video path:{video_path} does not exist, please check")
    return video_path


def temporalbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc['question']


def temporalbench_process_results(doc, results):
    pred = results[0]
    data_dict = {"item": doc, "pred": pred}
    

    return {"temporalbench_score": data_dict}


def temporalbench_caption_aggregate_results(results):
    from sentence_transformers import SentenceTransformer, util
    preds = []
    for data in results:
        preds.append(
            {
                'idx': data['item']['idx'],
                'response': data['pred']
            }
        )
    id2question = {}
    for data in results:
        id2question[data['item']['idx']] = data['item']
        
    gt_list = [ id2question[pred['idx']]['GT'] for pred in preds]
    ref_list = [ pred['response'] for pred in preds]
    
    model_name='all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    model = model.to('cuda:0')
    
    # Combine ref and gt lists into a big batch for encoding
    combined_sentences = ref_list + gt_list
    
    # Encode the batch with CUDA
    embeddings = model.encode(combined_sentences, convert_to_tensor=True, device='cuda')
    
    # Split embeddings into ref and gt parts
    ref_embeddings = embeddings[:len(ref_list)]
    gt_embeddings = embeddings[len(ref_list):]
    
    # Calculate cosine similarities between each ref and gt pair
    cosine_scores = util.cos_sim(ref_embeddings, gt_embeddings).diagonal()
    
    # Calculate the average similarity
    avg_similarity = cosine_scores.mean().item()*100
    
    return avg_similarity
    
def temporalbench_aggregate_results(results):
    preds = []
    for data in results:
        preds.append(
            {
                'idx': data['item']['idx'],
                'response': data['pred']
            }
        )
        
    id2question = {}
    for data in results:
        id2question[data['item']['idx']] = data['item']
    correct_count = 0
    multiple_binary_qa_correct = {}
    binary_qa_per_dataset = {}
    multiple_binary_qa_per_dataset = {}
    
    if 'category' in data['item'] and data['item']['category']!='':
      binary_qa_per_category = {}
      multiple_binary_qa_per_category = {}
    
    for pred in preds:
      
      # Binary QA Accuracy
      idx = pred['idx']
      gt = id2question[idx]['GT']
      predict_correct = gt.lower() == pred['response'][0].lower()
      if predict_correct:
        correct_count += 1
        
      # Multiple Binary QA Accuracy
      video_name = id2question[idx]['video_name']
      if video_name not in multiple_binary_qa_correct:
        multiple_binary_qa_correct[video_name] = True
      if not predict_correct:
            multiple_binary_qa_correct[video_name]= False
      
      # Per dataset Performance
      dataset = id2question[idx]['dataset']
      if dataset not in binary_qa_per_dataset:
        binary_qa_per_dataset[dataset] = []
        multiple_binary_qa_per_dataset[dataset] = {}
      binary_qa_per_dataset[dataset].append(predict_correct)
      if video_name not in multiple_binary_qa_per_dataset[dataset]:
        multiple_binary_qa_per_dataset[dataset][video_name] = True
      if not predict_correct:
        multiple_binary_qa_per_dataset[dataset][video_name] = False
      
      # Per category Performance
      if 'category' in data['item'] and data['item']['category']!='':
        category = id2question[idx]['category']
        if category not in binary_qa_per_category:
          binary_qa_per_category[category] = []
          multiple_binary_qa_per_category[category] = {}
        binary_qa_per_category[category].append(predict_correct)
        if video_name not in multiple_binary_qa_per_category[category]:
          multiple_binary_qa_per_category[category][video_name] = True
        if not predict_correct:
          multiple_binary_qa_per_category[category][video_name] = False
      
      
    # Print the results
    # try:
    width_dataset = 40   # for dataset names
    width_counts = 15    # for correct/total counts
    width_percentage = 1 # for percentages
    loginfo = ''
    loginfo +='*' * 20
    Binary_accuracy =  correct_count/len(preds) * 100
    loginfo +='\n'
    loginfo += f"{'Binary Accuracy:':<{width_dataset}} {correct_count}/{len(preds):<{width_counts}} {Binary_accuracy:>{width_percentage}.2f}%"
    mba_correct = sum([1 for v in multiple_binary_qa_correct.values() if v])
    Multiple_Binary_accuracy = mba_correct/len(multiple_binary_qa_correct) * 100
    loginfo +='\n'
    loginfo +=f"{'Multiple Binary Accuracy:':<{width_dataset}} {mba_correct}/{len(multiple_binary_qa_correct):<{width_counts}} {Multiple_Binary_accuracy:>{width_percentage}.2f}%"
# Print header
    loginfo +='\n'
    loginfo +='+'*110
    loginfo +='\n'
    loginfo += f"|+++ {'Dataset':<{width_dataset}}Binary Accuracy  {'':<{7}}  {'':>{width_percentage}} "
    f"||| Multiple Binary Accuracy {'':<{width_counts}}  {'':>{width_percentage}}"
    loginfo +='\n'
    loginfo +='+'*110
    for dataset, binary_qa in binary_qa_per_dataset.items():
        mba_correct = sum([1 for v in multiple_binary_qa_per_dataset[dataset].values() if v])
        loginfo +='\n'
        loginfo +=f"|--- {dataset + ' ':<{width_dataset}} {sum(binary_qa)}/{len(binary_qa):<{width_counts}} {sum(binary_qa)/len(binary_qa) * 100:>{width_percentage}.2f}% ||| {mba_correct}/{len(multiple_binary_qa_per_dataset[dataset]):<{width_counts}} {mba_correct/len(multiple_binary_qa_per_dataset[dataset]) * 100:>{width_percentage}.2f}%"
    
    if 'category' in data['item'] and data['item']['category']!='':
        loginfo +='\n'
        loginfo +='+'*110
        loginfo +='\n'
        loginfo += f"|-- {'Category':<{width_dataset}}Binary Accuracy  {'':<{7}}  {'':>{width_percentage}} "
        f"||| Multiple Binary Accuracy {'':<{width_counts}}  {'':>{width_percentage}}"
        loginfo +='\n'
        loginfo +='+'*110
        category_mapping = {
                                1: 'Action Order',
                                2: 'Action Frequency',
                                3: 'Action Type',
                                4: 'Motion Magnitude',
                                5: 'Motion Direction/Orientation',
                                6: 'Action Effector',
                                8: 'Event Order',
                                7: 'Others',
                        }
        for category_index, category in category_mapping.items():
          if category in binary_qa_per_category:
            binary_qa = binary_qa_per_category[category]
            mba_correct = sum([1 for v in multiple_binary_qa_per_category[category].values() if v])
            loginfo +='\n'
            loginfo +=f"|--- {category + ' ':<{width_dataset}} {sum(binary_qa)}/{len(binary_qa):<{width_counts}} {sum(binary_qa)/len(binary_qa) * 100:>{width_percentage}.2f}% "f"||| {mba_correct}/{len(multiple_binary_qa_per_category[category]):<{width_counts}} {mba_correct/len(multiple_binary_qa_per_category[category]) * 100:>{width_percentage}.2f}%"
    eval_logger.info(loginfo)   
    return Binary_accuracy, Multiple_Binary_accuracy