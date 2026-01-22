#!/usr/bin/env python3
"""
Simple Sliding Puzzle evaluator - works with output JSON only.
Shows prediction details without needing ground truth dataset.

Usage:
    python evaluate_sliding_simple.py logs_output.json
"""

import json
import re
import sys
from typing import List, Dict


def parse_predicted_moves(result_text: str) -> tuple:
    """
    Parse predicted moves from output text.
    Returns: (pred_moves, tag_found, raw_json_str)
    """
    pred_moves = []
    tag_found = None
    raw_json_str = None
    
    # Try to find <ANSWER_JSON> or <ANSWER JSON> or <ANSWERinch> (model errors)
    patterns = [
        (r"<ANSWER[_\s]JSON>\s*(\[.*?\])\s*</ANSWER[_\s]JSON>", "ANSWER_JSON"),
        (r"<ANSWERinch>\s*(\[.*?\])", "ANSWERinch (ERROR TAG)"),
        (r"<ANSWER[_\s]json>\s*(\[.*?\])", "ANSWER_json (lowercase)"),
    ]
    
    for pattern, tag_name in patterns:
        matches = list(re.finditer(pattern, result_text, re.DOTALL | re.IGNORECASE))
        if matches:
            tag_found = tag_name
            raw_json_str = matches[-1].group(1)
            try:
                moves_data = json.loads(raw_json_str)
                pred_moves = [str(m).strip().lower() for m in moves_data]
                break
            except:
                pass
    
    return pred_moves, tag_found, raw_json_str


def analyze_single_sample(doc_id: int, result_raw: str) -> Dict:
    """Analyze a single sliding puzzle output."""
    
    # Parse the model output JSON
    result_text = ""
    images = []
    
    if isinstance(result_raw, str):
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
        result_text = str(result_raw)
    
    # Parse predicted moves
    pred_moves, tag_found, raw_json = parse_predicted_moves(result_text)
    
    # Check for issues
    issues = []
    if not tag_found:
        issues.append("NO_TAG_FOUND")
    elif "ERROR" in tag_found or "inch" in tag_found or "lowercase" in tag_found:
        issues.append(f"WRONG_TAG: {tag_found}")
    
    if not pred_moves:
        issues.append("NO_MOVES_PARSED")
    
    if len(pred_moves) > 50:
        issues.append(f"TOO_MANY_MOVES: {len(pred_moves)}")
    
    # Analyze text patterns
    text_preview = result_text[:200] if len(result_text) > 200 else result_text
    
    return {
        "doc_id": doc_id,
        "pred_moves": pred_moves,
        "num_pred_moves": len(pred_moves),
        "num_images": len(images),
        "tag_found": tag_found,
        "issues": issues,
        "text_preview": text_preview,
        "has_repetition": len(pred_moves) > 20 and len(set(pred_moves)) < 5,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_sliding_simple.py <logs_output.json>")
        print("\nOr provide the JSON content directly via stdin")
        sys.exit(1)
    
    # Load responses
    if sys.argv[1] == "-":
        # Read from stdin
        responses = json.load(sys.stdin)
    else:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            responses = json.load(f)
    
    # Analyze each sample
    results = []
    task_prefix = "uni_mmmu_sliding54_visual_cot___train___"
    
    for key, value in sorted(responses.items()):
        if not key.startswith(task_prefix):
            continue
        
        # Extract doc_id
        doc_id_str = key.replace(task_prefix, "")
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            continue
        
        result = analyze_single_sample(doc_id, value)
        results.append(result)
    
    # Print summary
    if results:
        print("\n" + "="*100)
        print("SLIDING PUZZLE OUTPUT ANALYSIS")
        print("="*100)
        print(f"Total samples: {len(results)}\n")
        
        # Count issues
        issue_counts = {}
        normal_count = 0
        error_samples = []
        
        for r in results:
            if r["issues"]:
                for issue in r["issues"]:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                error_samples.append(r)
            else:
                normal_count += 1
        
        print(f"✓ Normal outputs: {normal_count}/{len(results)}")
        print(f"✗ Problematic outputs: {len(error_samples)}/{len(results)}\n")
        
        if issue_counts:
            print("Issue breakdown:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                print(f"  - {issue}: {count} samples")
            print()
        
        # Show per-sample details
        print("-" * 100)
        print(f"{'Doc':<5} {'Status':<15} {'Moves':<8} {'Images':<8} {'Tag':<20} {'Issues'}")
        print("-" * 100)
        
        for r in results:
            status = "✓ NORMAL" if not r["issues"] else "✗ ERROR"
            status_color = status
            
            if r["has_repetition"]:
                status_color += " [REPETITION]"
            
            issues_str = ", ".join(r["issues"]) if r["issues"] else "-"
            
            print(f"{r['doc_id']:<5d} {status_color:<15} {r['num_pred_moves']:<8d} "
                  f"{r['num_images']:<8d} {(r['tag_found'] or 'NONE'):<20} {issues_str}")
        
        print("-" * 100)
        
        # Show error details
        if error_samples:
            print("\nERROR DETAILS:")
            print("="*100)
            for r in error_samples:
                print(f"\nDoc {r['doc_id']}:")
                print(f"  Issues: {', '.join(r['issues'])}")
                print(f"  Tag found: {r['tag_found']}")
                print(f"  Predicted moves ({r['num_pred_moves']}): {r['pred_moves'][:10]}...")
                print(f"  Text preview: {r['text_preview']}")
                print()
        
        # Statistics
        print("\nSTATISTICS:")
        print("="*100)
        move_counts = [r['num_pred_moves'] for r in results if not r['issues']]
        if move_counts:
            avg_moves = sum(move_counts) / len(move_counts)
            min_moves = min(move_counts)
            max_moves = max(move_counts)
            print(f"Move count (normal samples): min={min_moves}, max={max_moves}, avg={avg_moves:.1f}")
        
        image_counts = [r['num_images'] for r in results]
        avg_images = sum(image_counts) / len(image_counts)
        print(f"Image count: min={min(image_counts)}, max={max(image_counts)}, avg={avg_images:.1f}")
        
        print("\nNote: To compute accuracy, you need the ground truth dataset.")
        print("Use evaluate_sliding_output.py with --dataset_file for full evaluation.")
    
    else:
        print("No sliding samples found in the input!")


if __name__ == "__main__":
    main()
