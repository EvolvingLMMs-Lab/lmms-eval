import os
import ast
import yaml
import json
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from loguru import logger as eval_logger
from lmms_eval.tasks.capability.prompt import Prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
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
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")
HF_HOME = os.path.expanduser(HF_HOME)
cache_dir = os.path.join(HF_HOME, config["dataset_kwargs"]["cache_dir"])


def capability_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    data_type = doc['data_type']
    file_path = doc['file_path'][5:]
    file_path = os.path.join(cache_dir, file_path)
    if not os.path.exists(file_path):
        eval_logger.error(f"File path: {file_path} does not exist, please check.")
    
    if data_type == 'image':
        return [Image.open(file_path).convert('RGB')]
    else:   # video
        return [file_path]


def capability_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    data_type = doc['data_type']
    return lmms_eval_specific_kwargs[f"{data_type}_prompt"]


def capability_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case capability_perception_score), value: metric value
    """
    if isinstance(doc["annotation"], dict):
        annotation = {k: v for k, v in doc["annotation"].items() if v is not None}
    else:
        annotation = doc["annotation"]

    response = {
        "file_id": doc["file_id"],
        "caption": results[0].strip(),
        "annotation": annotation,
        "task": doc["task"],
    }
    return {
        "capability_inference_result": response,
        "capability_precision": response,
        "capability_recall": response,
        "capability_f1_score": response,

    }


def capability_aggregate_inference_result(results, args):
    task = results[0]['task']
    if 'eval_save_root' in config['metadata'] and config['metadata']['eval_save_root'] is not None:
        save_path = os.path.join(config['metadata']['eval_save_root'], f"inference/{task}.jsonl")
    else:
        suffix = args.model if args.log_samples_suffix == "model_outputs" else args.log_samples_suffix
        save_path = generate_submission_file(
            file_name=f"{task}.jsonl",
            args=args,
            subpath=f"capability_results/{suffix}/inference"
        )

    # delete the invalid evaluation results as lmms-eval do not support auto-resume inference
    # to ensure re-run evaluation if re-run inference
    eval_save_path = os.path.join(os.path.dirname(save_path), f"../evaluation/{task}.jsonl")
    if os.path.exists(eval_save_path):
        os.remove(eval_save_path)
    
    with open(save_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    return None


def capability_aggregate_results(results, args):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    # results: [{"file_id": doc["file_id"], "caption": results[0].strip(), "annotation": doc["annotation"], "task": doc["task"]},]
    task = results[0]['task']
    if 'eval_save_root' in config['metadata'] and config['metadata']['eval_save_root'] is not None:
        save_path = os.path.join(config['metadata']['eval_save_root'], f"evaluation/{task}.jsonl")
    else:
        suffix = args.model if args.log_samples_suffix == "model_outputs" else args.log_samples_suffix
        save_path = generate_submission_file(
            file_name=f"{task}.jsonl",
            args=args,
            subpath=f"capability_results/{suffix}/evaluation"
        )
    eval_model = config['metadata']['eval_model_name']
    num_process = config['metadata']['eval_num_process']
    max_allow_missing = config['metadata']['eval_max_allow_missing']
    max_retry_times = config['metadata']['eval_max_retry_times']
    auto_resume = config['metadata']['eval_auto_resume']
    strict_match = config['metadata']['eval_strict_match']
    evaluator = Evaluator(
        task, results, save_path,
        eval_model, headers, num_process,
        max_allow_missing, max_retry_times,
        auto_resume, strict_match
    )
    score_dict = evaluator.evaluate_scores()
    metrics = evaluator.calculate_metric(score_dict)
    return metrics


def capability_aggregate_precision(results, args):
    metrics = capability_aggregate_results(results, args)
    task = results[0]['task']
    precision = metrics['precision']
    eval_logger.info(f"[{task}] precision: {precision:.1f}")
    return precision


def capability_aggregate_recall(results, args):
    metrics = capability_aggregate_results(results, args)
    task = results[0]['task']
    recall = metrics['recall']
    eval_logger.info(f"[{task}] recall: {recall:.1f}")
    return recall


def capability_aggregate_f1score(results, args):
    metrics = capability_aggregate_results(results, args)
    task = results[0]['task']
    f1_score = metrics['f1_score']
    eval_logger.info(f"[{task}] f1_score: {f1_score:.1f}")
    return f1_score


class Evaluator:
    def __init__(
            self, task, results, save_path,
            eval_model, headers, num_process=0,
            max_allow_missing=5, max_retry_times=10,
            auto_resume=True, strict_match=True,
    ):
        self.task = task
        self.results = results
        self.save_path = save_path
        self.eval_model = eval_model
        self.headers = headers
        self.num_process = num_process
        self.max_allow_missing = max_allow_missing
        self.max_retry_times = max_retry_times
        self.auto_resume = auto_resume
        self.strict_match = strict_match
        self.prompts = Prompts()

        self.post_validate_format_func = eval(f"self.post_validate_format_{task}")
        self.post_process_func = eval(f"self.post_process_{task}")

        self.file2anno = {r['file_id']: r['annotation'] for r in self.results}

    def post_validate_format_event(self, response, anno):
        # "{\"action\": \"copy provided action here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["event"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_event(self, response, anno):
        return response["score"]
    
    def post_validate_format_action(self, response, anno):
        # "{\"action\": \"copy provided action here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["action"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_action(self, response, anno):
        return response["score"]

    def post_validate_format_object_category(self, response, anno):
        # "{\"object_category\": \"copy provided object here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["object_category"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_object_category(self, response, anno):
        return response["score"]
    
    def post_validate_format_object_number(self, response, anno):
        # "{\"object_number\": \"copy the provided {object: number} here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if isinstance(response['object_number'], str):
            # assert response['object_number'].startswith("{") and response['object_number'].endswith("}")
            assert ':' in response['object_number']
            object_category, object_number = response['object_number'].lstrip('{').rstrip('}').split(":")
            object_number = int(object_number.strip())
        elif isinstance(response['object_number'], dict):
            object_category, object_number = list(response['object_number'].items())[0]
            object_number = int(object_number.strip())
        else:
            raise ValueError("Invalid object_number format")
        if self.strict_match:
            assert object_number == list(anno.values())[0]
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_object_number(self, response, anno):
        return response["score"]

    def post_validate_format_dynamic_object_number(self, response, anno):
        # "{\"object_number\": \"copy the provided {object: number} here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert 'response' in response
        for i, r in enumerate(response['response']):
            if isinstance(r['object_number'], str):
                # assert response['object_number'].startswith("{") and response['object_number'].endswith("}")
                assert ':' in r['object_number']
                object_category, object_number = r['object_number'].lstrip('{').rstrip('}').split(":")
                object_number = int(object_number.strip())
            elif isinstance(response['object_number'], dict):
                object_category, object_number = list(r['object_number'].items())[0]
                object_number = int(object_number.strip())
            else:
                raise ValueError("Invalid object_number format")
            if self.strict_match:
                assert object_number == list(anno.values())[i]
            if r["score"] in ["-1", "0", "1"]:
                r["score"] = int(r["score"])
            assert r["score"] in [1, 0, -1]

    def post_process_dynamic_object_number(self, response, anno):
        scores = []
        for r in response['response']:
            scores.append(r['score'])
        return scores

    def post_validate_format_object_color(self, response, anno):
        # "{\"object_color\": \"copy the provided {object: color} here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if isinstance(response['object_color'], str):
            # assert response['object_color'].startswith("{") and response['object_color'].endswith("}")
            assert ':' in response['object_color']
            unpacked = response['object_color'].lstrip('{').rstrip('}').split(":")
            if len(unpacked) > 2:
                object_category, object_color = ":".join(unpacked[:-1]), unpacked[-1]
            else:
                object_category, object_color = unpacked
            object_color = object_color.strip()
        elif isinstance(response['object_color'], dict):
            object_category, object_color = list(response['object_color'].items())[0]
            object_color = object_color.strip()
        else:
            raise ValueError("Invalid object_color format")
        if self.strict_match:
            assert object_color == list(anno.values())[0]
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_object_color(self, response, anno):
        return response["score"]

    def post_validate_format_spatial_relation(self, response, anno):
        # "{\"spatial_relation\": \"copy the provided spatial relationship here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["spatial_relation"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_spatial_relation(self, response, anno):
        return response["score"]

    def post_validate_format_scene(self, response, anno):
        # "{\"scene\": \"copy the provided scene here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["scene"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_scene(self, response, anno):
        return response["score"]

    def post_validate_format_camera_angle(self, response, anno):
        # "{\"pred\": \"put your predicted category here\", \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert "pred" in response
        if response["pred"] == "N/A" or "N/A" in response["pred"]:
            response["pred"] = ["N/A"]
        if isinstance(response["pred"], str):
            response["pred"] = ast.literal_eval(response['pred'])
        assert isinstance(response["pred"], list)
        for i in range(len(response["pred"])):
            if response["pred"][i] in self.prompts.camera_angle_category_explains:
                response["pred"][i] = response["pred"].split(":")[0].lower()
            assert response["pred"][i] == "N/A" or response["pred"][i] in self.prompts.camera_angle_categories
    
    def post_process_camera_angle(self, response, anno):
        if len(response["pred"]) == 1 and response["pred"][0] == "N/A":
            return 0
        elif anno in response["pred"]:
            return 1
        else:
            return -1

    def post_validate_format_camera_movement(self, response, anno):
        # "{\"pred\": \"put your predicted category here\", \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert "pred" in response
        if response["pred"] == "N/A" or "N/A" in response["pred"]:
            response["pred"] = ["N/A"]
        if isinstance(response["pred"], str):
            response["pred"] = ast.literal_eval(response['pred'])
        assert isinstance(response["pred"], list)
        for i in range(len(response["pred"])):
            if response["pred"][i] in self.prompts.camera_movement_category_explains:
                response["pred"][i] = response["pred"].split(":")[0].lower()
            assert response["pred"][i] == "N/A" or response["pred"][i] in self.prompts.camera_movement_categories
    
    def post_process_camera_movement(self, response, anno):
        if len(response["pred"]) == 1 and response["pred"][0] == "N/A":
            return 0
        elif anno in response["pred"]:
            return 1
        else:
            return -1

    def post_validate_format_OCR(self, response, anno):
        # "{\"OCR\": \"copy the provided real OCR text here\", \"score\": put your score here, \"reason\": \"give your reason here\"},\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response['OCR'].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]
        
    def post_process_OCR(self, response, anno):
        return response['score']

    def post_validate_format_style(self, response, anno):
        # "{\"pred\": \"put your predicted category here\", \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        assert "pred" in response
        if response["pred"] == "N/A" or "N/A" in response["pred"]:
            response["pred"] = ["N/A"]
        if isinstance(response["pred"], str):
            response["pred"] = ast.literal_eval(response['pred'])
        assert isinstance(response["pred"], list)
        for i in range(len(response["pred"])):
            if response["pred"][i] in self.prompts.style_category_explains:
                response["pred"][i] = response["pred"][i].split(":")[0].lower()
            assert response["pred"][i] == "N/A" or response["pred"][i] in self.prompts.style_categories

    def post_process_style(self, response, anno):
        if len(response["pred"]) == 1 and response["pred"][0] == "N/A":
            return 0
        elif anno in response["pred"]:
            return 1
        else:
            return -1
    
    def post_validate_format_character_identification(self, response, anno):
        # "{\"name\": \"copy the provided name here\", \"score\": \"put your score here\",  \"reason\": \"give your reason here\"}\n"\
        assert isinstance(response, dict)
        if self.strict_match:
            assert response["character_identification"].strip() == anno.strip()
        if response["score"] in ["-1", "0", "1"]:
            response["score"] = int(response["score"])
        assert response["score"] in [1, 0, -1]

    def post_process_character_identification(self, response, anno):
        return response["score"]
        
    def load_saved_records(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                saved_responses = [json.loads(l.strip('\n')) for l in f.readlines()]
        else:
            saved_responses = []
        return saved_responses

    def call_gpt(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            payload = {
                "model": self.eval_model,
                "messages": messages,
            }
            response = requests.post(API_URL, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            response = response.json()
        except Exception as e:
            eval_logger.info(f"Error calling {self.eval_model}: {e}")
            return None
        
        try:
            response_message = response["choices"][0]["message"]["content"].strip()
            return response_message
        except Exception as e:
            eval_logger.info(f"Error parsing {self.eval_model} response: {e}\nResponse: {response}")
            return None

    def call_and_parse_single_meaasge(self, file, system_prompt, user_prompt):
        response_message = self.call_gpt(system_prompt, user_prompt)
        if response_message is None:
            return None

        try:
            if '```json' in response_message:
                response_message = response_message.split('```json')[-1].split('```')[0].strip()
            if '```python' in response_message:
                response_message = response_message.split('```python')[-1].split('```')[0].strip()
            elif '```' in response_message:
                response_message = response_message.split('```')[1].strip()
            response = ast.literal_eval(response_message)
            return response
        except (SyntaxError, ValueError) as e:
            eval_logger.info(f"Invalid response format for {file}: {response_message}")
            return None

    def evaluate_sample_worker(self, args):
        file, anno, system_prompt, user_prompt = args
        if isinstance(user_prompt, list):
            response = {'response': []}
            for prompt in user_prompt:
                single_response = self.call_and_parse_single_meaasge(file, system_prompt, prompt)
                if single_response is None:
                    return None
                response['response'].append(single_response)
            
        else:
            response = self.call_and_parse_single_meaasge(file, system_prompt, user_prompt)
            if response is None:
                return None
        
        try:
            self.post_validate_format_func(response, anno)
        except Exception as e:
            eval_logger.info(f"Format validation failed for {file}: {e}, anno: {anno}, response: {response}")
            return None

        response['file_id'] = file
        return response

    def evaluate_scores(self):
        score_dict = {}
        # Load saved records for resuming evaluation
        if self.auto_resume:
            saved_responses = self.load_saved_records()
            eval_logger.info(f"[{self.task}] Loaded {len(saved_responses)} records")
        else:
            saved_responses = []
        
        buffer = []
        buffer_size = 100
        try:
            # Evaluate remaining
            for retry_count in range(self.max_retry_times + 1):
                saved_files = [r['file_id'] for r in saved_responses]
                if len(saved_files) == len(self.results):
                    break
                if len(self.results) - len(saved_files) <= self.max_allow_missing:
                    break

                remaining_results = [r for r in self.results if r['file_id'] not in saved_files]
                if retry_count != 0:
                    print(f"\nRetrying {retry_count} times")
                
                process_args = []
                for res in remaining_results:
                    file = res['file_id']
                    caption = res['caption']
                    anno = res['annotation']
                    system_prompt, user_prompt = self.prompts.get_prompts_by_task(self.task, caption, anno)
                    args = (file, anno, system_prompt, user_prompt)
                    process_args.append(args)
                
                if self.num_process == 0:
                    for args in tqdm(process_args, desc=f"Evaluating {self.task}"):
                        response = self.evaluate_sample_worker(args)
                        if response is not None:
                            with open(self.save_path, 'a') as f:
                                f.write(json.dumps(response) + '\n')
                            saved_responses.append(response)
                else:
                    with ThreadPoolExecutor(max_workers=self.num_process) as executor:
                        futures = {executor.submit(self.evaluate_sample_worker, arg): arg for arg in process_args}
                        buffer_counter = 0
                        for future in tqdm(as_completed(futures), total=len(remaining_results), desc=f"Evaluating {self.task}"):
                            result = future.result()
                            if result is not None:
                                buffer.append(json.dumps(result) + '\n')
                                buffer_counter += 1
                                if buffer_counter >= buffer_size:
                                    with open(self.save_path, 'a') as f:
                                        f.writelines(buffer)
                                    buffer.clear()
                                    buffer_counter = 0
                                
                                saved_responses.append(result)
                        
                        if len(buffer) > 0:
                            with open(self.save_path, 'a') as f:
                                f.writelines(buffer)
                            buffer.clear()

        finally:
            if len(buffer) > 0:
                with open(self.save_path, 'a') as f:
                    f.writelines(buffer)
                buffer.clear()

        
        for response in tqdm(saved_responses, desc=f"Calculating {self.task} scores"):
            file = response['file_id']
            score_dict[file] = self.post_process_func(response, self.file2anno[file])
            
        return score_dict

    def calculate_metric(self, score_dict):
        all_scores = []
        for file_id, scores in score_dict.items():
            if isinstance(scores, list):
                all_scores += scores
            else:
                all_scores.append(scores)
        all_scores = np.array(all_scores)
        sum_count = len(all_scores)
        hit_count = np.count_nonzero(all_scores != 0)
        correct_count = np.count_nonzero(all_scores == 1)
        precision = 0 if hit_count == 0 else 100 * correct_count / hit_count
        recall = 100 * correct_count / sum_count
        hit_rate = 100 * hit_count / sum_count
        f1_score = 0 if precision == 0 else 2 * precision * recall / (precision + recall)
        eval_logger.info(f"[{self.task}] all: {sum_count}, hit: {hit_count}, correct: {correct_count}")
        return {
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
            "f1_score": f1_score
        }
    

# Directly run this file to evaluate existing inference record
if __name__ == "__main__":
    results_dir = "logs/capability_results/llava_onevision_7b/inference"
    save_dir = "logs/capability_results/llava_onevision_7b/evaluation"
    os.makedirs(save_dir, exist_ok=True)

    tasks = ["object_category", "object_number", "object_color", "spatial_relation", 
             "scene", "camera_angle", "OCR", "style", "character_identification", 
             "dynamic_object_number", "action", "camera_movement", "event"]
    
    metrics = []
    for task in tasks:
        with open(os.path.join(results_dir, f"{task}.jsonl"), 'r') as f:
            result = [json.loads(l.strip()) for l in f.readlines()]
        save_path = os.path.join(save_dir, f"{task}.jsonl")
        eval_model = config['metadata']['eval_model_name']
        num_process = config['metadata']['eval_num_process']
        max_allow_missing = config['metadata']['eval_max_allow_missing']
        max_retry_times = config['metadata']['eval_max_retry_times']
        auto_resume = config['metadata']['eval_auto_resume']
        strict_match = config['metadata']['eval_strict_match']
        evaluator = Evaluator(
            task, result, save_path,
            eval_model, headers, num_process,
            max_allow_missing, max_retry_times,
            auto_resume, strict_match
        )
        score_dict = evaluator.evaluate_scores()
        metric = evaluator.calculate_metric(score_dict)
        metrics.append(metric)
        eval_logger.info(f"[{task}] " + ", ".join([f"{k}: {v:.1f}" for k, v in metric.items()]))
    
    # summarize metrics
    eval_logger.info("Summarized Results:")
    avg_precision = np.mean([m["precision"] for m in metrics])
    avg_recall = np.mean([m["recall"] for m in metrics])
    avg_hit_rate = np.mean([m["hit_rate"] for m in metrics])
    avg_f1_score = np.mean([m["f1_score"] for m in metrics])
    eval_logger.info(f"Average precision: {avg_precision:.3f}, recall: {avg_recall:.3f}, f1_score: {avg_f1_score:.3f}, hit_rate: {avg_hit_rate:.3f}")
