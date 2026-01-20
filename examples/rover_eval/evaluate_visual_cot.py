#!/usr/bin/env python3
"""
Visual CoT ROVER 评测示例脚本

用法:
    python examples/rover_eval/evaluate_visual_cot.py \
        --log_dir ./logs/bagel_visual_cot/ \
        --image_dir ./dataset/illusionbench/images/ \
        --output visual_cot_results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from lmms_eval.rover_eval import VisualCoTEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual CoT outputs using ROVER")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing JSON metadata files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing original question images")
    parser.add_argument("--output", type=str, default="visual_cot_results.csv", help="Output CSV file")
    parser.add_argument("--metrics", nargs="+", default=["ra", "al"], choices=["ra", "al"], help="Metrics to evaluate")
    parser.add_argument("--task_category", type=str, default=None, 
                       choices=["real_world", "mathematical", "stem", "puzzles", "chart_table", "spatial", "perception"],
                       help="Task category for customized prompts")
    parser.add_argument("--max_workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per evaluation")
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    image_dir = Path(args.image_dir)
    
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} not found")
        sys.exit(1)
    
    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} not found")
        sys.exit(1)
    
    # 收集所有 JSON 文件
    json_files = sorted(log_dir.glob("*_metadata.json"))
    print(f"Found {len(json_files)} JSON files in {log_dir}")
    
    if len(json_files) == 0:
        print("No JSON files found. Exiting.")
        sys.exit(1)
    
    # 准备评测数据
    json_paths = []
    original_images = []
    
    for json_file in json_files:
        with open(json_file) as f:
            try:
                data = json.load(f)
                doc_id = data.get("doc_id", "unknown")
                
                # 尝试多种图像文件格式
                original_img = None
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                    candidate = image_dir / f"{doc_id}{ext}"
                    if candidate.exists():
                        original_img = candidate
                        break
                
                if original_img is None:
                    print(f"Warning: Image for doc_id {doc_id} not found in {image_dir}, skipping")
                    continue
                
                json_paths.append(str(json_file))
                original_images.append(str(original_img))
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}, skipping")
                continue
    
    if len(json_paths) == 0:
        print("No valid samples to evaluate. Exiting.")
        sys.exit(1)
    
    print(f"Evaluating {len(json_paths)} samples")
    print(f"  Metrics: {args.metrics}")
    if args.task_category:
        print(f"  Task Category: {args.task_category}")
    
    # 初始化评测器
    evaluator = VisualCoTEvaluator(
        metrics=args.metrics,
        task_category=args.task_category,
        max_retries=args.max_retries
    )
    
    # 批量评测
    results = evaluator.evaluate_batch(
        json_paths=json_paths,
        original_images=original_images,
        max_workers=args.max_workers
    )
    
    # 保存结果
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to {args.output}")
    
    # 统计信息
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    
    if "ra" in args.metrics:
        valid_ra = df['ra_score'].notna()
        print(f"\nRA Evaluation:")
        print(f"  Valid: {valid_ra.sum()}/{len(df)}")
        print(f"  Average Score: {df.loc[valid_ra, 'ra_score'].mean():.2f}")
        print(f"  Score Distribution:")
        for score in sorted(df.loc[valid_ra, 'ra_score'].unique()):
            count = (df['ra_score'] == score).sum()
            print(f"    {int(score)}: {count} samples ({count/valid_ra.sum()*100:.1f}%)")
    
    if "al" in args.metrics:
        valid_al = df['al_score'].notna()
        print(f"\nAL Evaluation:")
        print(f"  Valid: {valid_al.sum()}/{len(df)}")
        print(f"  Average Score: {df.loc[valid_al, 'al_score'].mean():.2f}")
        print(f"  Score Distribution:")
        for score in sorted(df.loc[valid_al, 'al_score'].unique()):
            count = (df['al_score'] == score).sum()
            print(f"    {int(score)}: {count} samples ({count/valid_al.sum()*100:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    main()
