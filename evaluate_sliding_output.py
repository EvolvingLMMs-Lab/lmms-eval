#!/usr/bin/env python3
"""
Evaluate Sliding Puzzle outputs directly from model response cache.

Usage:
    python evaluate_sliding_output.py \
        --response_file ./logs/bagel_persistent_folder/bagel_response.json \
        --dataset_file /blob/lmms-eval-dataset/uni_mmmu_sliding54.parquet

Or provide the output JSON directly:
    python evaluate_sliding_output.py \
        --response_json '{"uni_mmmu_sliding54_visual_cot___train___0": "...", ...}' \
        --dataset_file /blob/lmms-eval-dataset/uni_mmmu_sliding54.parquet
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd


def parse_predicted_moves(result_text: str) -> List[str]:
    """Parse predicted moves from <ANSWER_JSON> tag."""
    pred_moves = []
    
    # Find all <ANSWER_JSON> matches (case insensitive)
    matches = list(re.finditer(
        r"<ANSWER[_\s]JSON>\s*(\[.*?\])\s*</ANSWER[_\s]JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    
    if matches:
        try:
            # Use the last match
            moves_data = json.loads(matches[-1].group(1))
            pred_moves = [str(m).strip().lower() for m in moves_data]
        except:
            pass
    
    return pred_moves


def evaluate_single_sample(
    result_raw: str,
    gt_moves: List[str],
    doc_id: int
) -> Dict:
    """Evaluate a single sliding puzzle sample."""
    
    # Parse the model output JSON
    result_text = ""
    images = []
    
    if isinstance(result_raw, dict):
        # Already parsed dictionary
        result_text = result_raw.get("text", "")
        images = result_raw.get("images", [])
    elif isinstance(result_raw, str):
        # String that needs parsing
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
                images = parsed_result.get("images", [])
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            result_text = result_raw
    else:
        # Fallback
        result_text = str(result_raw)
    
    # Parse predicted moves
    pred_moves = parse_predicted_moves(result_text)
    
    # Ground truth moves (ensure lowercase)
    gt_moves = [str(m).lower() for m in gt_moves]
    
    # Calculate metrics
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )
    
    return {
        "doc_id": doc_id,
        "pred_moves": pred_moves,
        "gt_moves": gt_moves,
        "num_pred_moves": len(pred_moves),
        "num_gt_moves": len(gt_moves),
        "num_images": len(images),
        "text_exact": text_exact,
        "text_frame_acc": text_frame_acc,
        "images": images[:3] if len(images) > 3 else images,  # Show first 3
    }


def load_ground_truth(dataset_file: str) -> Dict[int, List[str]]:
    """Load ground truth from parquet file."""
    print(f"Loading ground truth from: {dataset_file}")
    
    if not Path(dataset_file).exists():
        print(f"Warning: Dataset file not found: {dataset_file}")
        return {}
    
    try:
        df = pd.read_parquet(dataset_file)
        gt_dict = {}
        
        for idx, row in df.iterrows():
            steps_words = row.get("steps_words", "[]")
            if isinstance(steps_words, str):
                gt_moves = json.loads(steps_words)
            else:
                gt_moves = steps_words
            gt_dict[idx] = [str(m).lower() for m in gt_moves]
        
        print(f"Loaded {len(gt_dict)} ground truth samples")
        return gt_dict
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sliding Puzzle outputs")
    parser.add_argument("--response_file", type=str, help="Path to response JSON file")
    parser.add_argument("--response_json", type=str, help="Response JSON string directly")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to dataset parquet file")
    parser.add_argument("--output", type=str, default="sliding_eval_results.json", help="Output file for detailed results")
    
    args = parser.parse_args()
    
    # Load responses
    if args.response_file:
        print(f"Loading responses from: {args.response_file}")
        with open(args.response_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Support different formats
        if isinstance(raw_data, dict):
            # Format: {"doc_id_key": "{json_string}", ...}
            responses = []
            for key, value in raw_data.items():
                # Extract doc_id from key (e.g., "uni_mmmu_sliding54_visual_cot___train___0" -> 0)
                parts = key.split("___")
                doc_id = int(parts[-1]) if parts else 0
                
                # Parse the JSON string value
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            parsed["doc_id"] = doc_id
                            responses.append(parsed)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse response for {key}")
                        continue
            print(f"Loaded {len(responses)} responses from dictionary format")
        elif isinstance(raw_data, list):
            # Format: [{"doc_id": ..., ...}, ...]
            responses = raw_data
            print(f"Loaded {len(responses)} responses from list format")
        else:
            raise ValueError(f"Unsupported response file format: {type(raw_data)}")
    elif args.response_json:
        responses = json.loads(args.response_json)
    else:
        print("Error: Must provide either --response_file or --response_json")
        return
    
    # Load ground truth
    gt_dict = load_ground_truth(args.dataset_file)
    
    # Evaluate each sample
    results = []
    
    for response in responses:
        # Get doc_id from response
        doc_id = response.get("doc_id")
        if doc_id is None:
            print(f"Warning: Response missing doc_id: {response}")
            continue
        
        # Get ground truth
        if doc_id not in gt_dict:
            print(f"Warning: No ground truth for doc_id={doc_id}")
            continue
        
        gt_moves = gt_dict[doc_id]
        
        # Evaluate
        result = evaluate_single_sample(response, gt_moves, doc_id)
        results.append(result)
    
    # Calculate overall metrics
    if results:
        avg_exact = sum(r["text_exact"] for r in results) / len(results)
        avg_frame_acc = sum(r["text_frame_acc"] for r in results) / len(results)
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Total samples evaluated: {len(results)}")
        print(f"sliding_text_exact:      {avg_exact:.4f} ({avg_exact*100:.2f}%)")
        print(f"sliding_text_frame_acc:  {avg_frame_acc:.4f} ({avg_frame_acc*100:.2f}%)")
        print("="*80)
        
        # Show per-sample details
        print("\nPer-sample results:")
        print("-" * 80)
        correct_count = 0
        for r in results:
            status = "✓ CORRECT" if r["text_exact"] == 1 else "✗ WRONG"
            if r["text_exact"] == 1:
                correct_count += 1
            
            print(f"Doc {r['doc_id']:2d}: {status} | "
                  f"Pred: {r['num_pred_moves']} moves, GT: {r['num_gt_moves']} moves | "
                  f"Frame Acc: {r['text_frame_acc']:.2f}")
            
            if r["text_exact"] == 0:
                print(f"         Predicted: {r['pred_moves']}")
                print(f"         Ground truth: {r['gt_moves']}")
        
        print("-" * 80)
        print(f"Correct: {correct_count}/{len(results)}")
        
        # Save detailed results
        output_data = {
            "summary": {
                "total_samples": len(results),
                "sliding_text_exact": avg_exact,
                "sliding_text_frame_acc": avg_frame_acc,
                "correct_samples": correct_count,
            },
            "per_sample": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {args.output}")
    
    else:
        print("No samples evaluated!")


if __name__ == "__main__":
    main()
