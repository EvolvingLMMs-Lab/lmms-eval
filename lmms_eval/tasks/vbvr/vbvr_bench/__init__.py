"""
VBVR-Bench
A rule-based evaluation toolkit for assessing video generation models on 100 visual reasoning tasks. Each task is evaluated by a dedicated rule-based evaluator.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import importlib

from .utils import (
    load_json, save_json, get_video_frames, get_frame_count,
    load_gt_metadata, extract_task_info_from_path
)
from .evaluators import is_out_of_domain, get_split, OUT_OF_DOMAIN_PREFIXES


class VBVRBench:
    """
    Main evaluation class for VBVR-Bench.
    Evaluates video generation models on 100 visual reasoning tasks using rule-based metrics.
    
    Results are reported by:
    - In_Domain: In-domain test set (50 tasks)
    - Out_of_Domain: Out-of-domain test set (50 tasks)
    """
    
    def __init__(
        self,
        gt_base_path: str,
        output_path: str = './evaluation_results/',
        device: str = 'cuda',
        rules_path: str = None
    ):
        """
        Initialize VBVRBench.
        
        Args:
            gt_base_path: Base path to ground truth videos and metadata
            output_path: Directory to save evaluation results
            device: Device for computation ('cuda' or 'cpu')
            rules_path: Path to task rules (txt files converted from PDFs)
        """
        self.gt_base_path = gt_base_path
        self.output_path = output_path
        self.device = device
        self.rules_path = rules_path
        
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load task definitions
        self.tasks_info = self._load_tasks_info()
        
    def _load_tasks_info(self) -> Dict:
        """Load task information from tasks.json if available."""
        tasks_json_path = os.path.join(self.gt_base_path, 'tasks.json')
        if os.path.exists(tasks_json_path):
            return load_json(tasks_json_path)
        return {"In_Domain": [], "Out_of_Domain": []}
    
    def get_all_tasks(self) -> List[str]:
        """Get list of all task names."""
        all_tasks = []
        for split in ['In_Domain', 'Out_of_Domain']:
            if split in self.tasks_info:
                all_tasks.extend(self.tasks_info[split])
        return list(set(all_tasks))
    
    def get_split_tasks(self, split: str) -> List[str]:
        """Get tasks for a specific split."""
        return self.tasks_info.get(split, [])
    
    def build_evaluation_info(
        self,
        videos_path: str,
        task_list: Optional[List[str]] = None,
        split: Optional[str] = None
    ) -> List[Dict]:
        """
        Build evaluation information for all videos to be evaluated.
        
        Args:
            videos_path: Path to the directory containing videos to evaluate
                        Structure: {videos_path}/{split}/{task_name}/{idx}.mp4
            task_list: Optional list of specific tasks to evaluate
            split: Optional specific split ('In-Domain_50' or 'Out-of-Domain_50')
            
        Returns:
            List of evaluation info dictionaries
        """
        eval_info_list = []
        
        if split:
            splits = [split]
        else:
            splits = ['In-Domain_50', 'Out-of-Domain_50']
        
        for current_split in splits:
            split_path = os.path.join(videos_path, current_split)
            if not os.path.exists(split_path):
                continue
                
            tasks = task_list if task_list else os.listdir(split_path)
            
            for task_name in tasks:
                task_path = os.path.join(split_path, task_name)
                if not os.path.isdir(task_path):
                    continue
                
                # Get all video files for this task
                video_files = sorted([
                    f for f in os.listdir(task_path) 
                    if f.endswith('.mp4')
                ])
                
                for video_file in video_files:
                    video_idx = video_file.replace('.mp4', '')
                    
                    # Build ground truth paths
                    gt_task_path = os.path.join(
                        self.gt_base_path, current_split, task_name, video_idx
                    )
                    
                    # Determine logical split (In_Domain vs Out_of_Domain)
                    logical_split = 'Out_of_Domain' if is_out_of_domain(task_name) else 'In_Domain'
                    
                    eval_info = {
                        'split': logical_split,
                        'file_split': current_split,  # Original folder name
                        'task_name': task_name,
                        'video_idx': video_idx,
                        'video_path': os.path.join(task_path, video_file),
                        'gt_path': gt_task_path,
                        'gt_video_path': os.path.join(gt_task_path, 'ground_truth.mp4'),
                        'gt_first_frame': os.path.join(gt_task_path, 'first_frame.png'),
                        'gt_final_frame': os.path.join(gt_task_path, 'final_frame.png'),
                        'prompt_path': os.path.join(gt_task_path, 'prompt.txt'),
                    }
                    
                    # Load prompt if available
                    if os.path.exists(eval_info['prompt_path']):
                        with open(eval_info['prompt_path'], 'r') as f:
                            eval_info['prompt'] = f.read().strip()
                    
                    eval_info_list.append(eval_info)
        
        return eval_info_list
    
    def evaluate(
        self,
        videos_path: str,
        name: Optional[str] = None,
        task_list: Optional[List[str]] = None,
        split: Optional[str] = None,
        save_detailed: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation on videos.
        
        Args:
            videos_path: Path to videos to evaluate
            name: Name for this evaluation run
            task_list: Optional list of specific tasks to evaluate
            split: Optional specific split to evaluate
            save_detailed: Whether to save detailed per-video results
            **kwargs: Additional arguments passed to evaluators
            
        Returns:
            Dictionary containing evaluation results organized by:
            - overall: Combined scores
            - In_Domain: In-domain test set scores (50 tasks)
            - Out_of_Domain: Out-of-domain test set scores (50 tasks)
            - by_task: Per-task breakdown
            - detailed: Per-video results
        """
        if name is None:
            name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        
        # Build evaluation info
        eval_info_list = self.build_evaluation_info(videos_path, task_list, split)
        
        print(f"Found {len(eval_info_list)} videos to evaluate")
        
        # Import evaluator
        from .evaluators import get_evaluator
        
        # Initialize results structure
        results = {
            'overall': {},
            'In_Domain': {'scores': [], 'by_task': {}},
            'Out_of_Domain': {'scores': [], 'by_task': {}},
            'by_task': {},
            'detailed': []
        }
        
        # Process each video
        for i, eval_info in enumerate(eval_info_list):
            task_name = eval_info['task_name']
            split_name = eval_info['split']  # Now uses In_Domain/Out_of_Domain
            
            if (i + 1) % 50 == 0:
                print(f"  Processing {i + 1}/{len(eval_info_list)}...")
            
            # Get appropriate evaluator for this task
            evaluator = get_evaluator(task_name, self.device)
            
            # Run evaluation
            try:
                video_result = evaluator.evaluate(eval_info, **kwargs)
            except Exception as e:
                print(f"Error evaluating {eval_info['video_path']}: {e}")
                video_result = {
                    'score': 0.0,
                    'error': str(e),
                    'dimensions': {}
                }
            
            # Add metadata
            video_result['video_path'] = eval_info['video_path']
            video_result['task_name'] = task_name
            video_result['split'] = split_name
            video_result['video_idx'] = eval_info['video_idx']
            
            results['detailed'].append(video_result)
            
            # Aggregate by split
            if split_name not in results:
                results[split_name] = {'scores': [], 'by_task': {}}
            results[split_name]['scores'].append(video_result['score'])
            
            # Aggregate by task within split
            if task_name not in results[split_name]['by_task']:
                results[split_name]['by_task'][task_name] = {
                    'scores': [],
                    'dimensions': {}
                }
            results[split_name]['by_task'][task_name]['scores'].append(video_result['score'])
            
            # Aggregate dimensions
            for dim_name, dim_score in video_result.get('dimensions', {}).items():
                if dim_name not in results[split_name]['by_task'][task_name]['dimensions']:
                    results[split_name]['by_task'][task_name]['dimensions'][dim_name] = []
                results[split_name]['by_task'][task_name]['dimensions'][dim_name].append(dim_score)
            
            # Also maintain overall by_task for compatibility
            if task_name not in results['by_task']:
                results['by_task'][task_name] = {
                    'scores': [],
                    'split': split_name,
                    'dimensions': {}
                }
            results['by_task'][task_name]['scores'].append(video_result['score'])
        
        # Calculate averages
        self._calculate_averages(results)
        
        # Calculate category scores
        self._calculate_category_scores(results)
        
        # Save results
        output_name = os.path.join(self.output_path, f'{name}_eval_results.json')
        save_json(results, output_name)
        print(f"Results saved to {output_name}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _calculate_averages(self, results: Dict):
        """Calculate average scores at all levels."""
        # Average for each split
        for split in ['In_Domain', 'Out_of_Domain']:
            if split in results and results[split]['scores']:
                results[split]['mean_score'] = sum(results[split]['scores']) / len(results[split]['scores'])
                results[split]['num_videos'] = len(results[split]['scores'])
                
                # Average per task within split
                for task_name, task_data in results[split]['by_task'].items():
                    task_data['mean_score'] = sum(task_data['scores']) / len(task_data['scores'])
                    
                    # Average dimensions
                    for dim_name, dim_scores in task_data['dimensions'].items():
                        task_data['dimensions'][dim_name] = sum(dim_scores) / len(dim_scores)
            else:
                if split not in results:
                    results[split] = {'scores': [], 'by_task': {}}
                results[split]['mean_score'] = 0.0
                results[split]['num_videos'] = 0
        
        # Average for overall by_task
        for task_name, task_data in results['by_task'].items():
            task_data['mean_score'] = sum(task_data['scores']) / len(task_data['scores']) if task_data['scores'] else 0
        
        # Overall average (weighted by number of videos)
        total_videos = results['In_Domain']['num_videos'] + results['Out_of_Domain']['num_videos']
        if total_videos > 0:
            results['overall']['mean_score'] = (
                results['In_Domain']['mean_score'] * results['In_Domain']['num_videos'] +
                results['Out_of_Domain']['mean_score'] * results['Out_of_Domain']['num_videos']
            ) / total_videos
            results['overall']['num_videos'] = total_videos
        else:
            results['overall']['mean_score'] = 0.0
            results['overall']['num_videos'] = 0
        
        # Store individual split scores for easy access
        results['overall']['In_Domain_score'] = results['In_Domain']['mean_score']
        results['overall']['Out_of_Domain_score'] = results['Out_of_Domain']['mean_score']
    
    def _calculate_category_scores(self, results: Dict):
        """Calculate scores by task category."""
        from .evaluators import get_task_category
        
        # Initialize category results
        results['by_category'] = {}
        
        # Aggregate scores by category
        for task_name, task_data in results['by_task'].items():
            category = get_task_category(task_name)
            
            if category not in results['by_category']:
                results['by_category'][category] = {
                    'scores': [],
                    'tasks': [],
                    'num_tasks': 0
                }
            
            results['by_category'][category]['scores'].extend(task_data['scores'])
            results['by_category'][category]['tasks'].append(task_name)
            results['by_category'][category]['num_tasks'] += 1
        
        # Calculate averages for each category
        for category, cat_data in results['by_category'].items():
            if cat_data['scores']:
                cat_data['mean_score'] = sum(cat_data['scores']) / len(cat_data['scores'])
                cat_data['num_videos'] = len(cat_data['scores'])
            else:
                cat_data['mean_score'] = 0.0
                cat_data['num_videos'] = 0
        
        # Store in overall
        results['overall']['by_category'] = {
            cat: data['mean_score'] for cat, data in results['by_category'].items()
        }
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("VBVR-Bench Evaluation Summary")
        print("=" * 70)
        print(f"Total videos evaluated: {results['overall']['num_videos']}")
        print()
        
        # Main results: In_Domain and Out_of_Domain averages
        print("┌" + "─" * 68 + "┐")
        print("│" + " " * 20 + "MAIN RESULTS" + " " * 36 + "│")
        print("├" + "─" * 34 + "┬" + "─" * 33 + "┤")
        in_domain_str = f"│  In-Domain (50 tasks):         │  {results['In_Domain']['mean_score']:.4f}  ({results['In_Domain']['num_videos']} videos)"
        print(in_domain_str + " " * (69 - len(in_domain_str)) + "│")
        ood_str = f"│  Out-of-Domain (50 tasks):     │  {results['Out_of_Domain']['mean_score']:.4f}  ({results['Out_of_Domain']['num_videos']} videos)"
        print(ood_str + " " * (69 - len(ood_str)) + "│")
        print("├" + "─" * 34 + "┼" + "─" * 33 + "┤")
        overall_str = f"│  Overall Average:              │  {results['overall']['mean_score']:.4f}"
        print(overall_str + " " * (69 - len(overall_str)) + "│")
        print("└" + "─" * 34 + "┴" + "─" * 33 + "┘")
        print()
        
        # Print category scores
        if 'by_category' in results and results['by_category']:
            print("┌" + "─" * 68 + "┐")
            print("│" + " " * 18 + "SCORES BY CATEGORY" + " " * 32 + "│")
            print("├" + "─" * 34 + "┬" + "─" * 33 + "┤")
            
            # Sort categories by score
            cat_scores = [(cat, data['mean_score'], data['num_tasks']) 
                         for cat, data in results['by_category'].items()]
            cat_scores.sort(key=lambda x: x[1], reverse=True)
            
            for cat, score, num_tasks in cat_scores:
                cat_str = f"│  {cat:<30} │  {score:.4f}  ({num_tasks} tasks)"
                print(cat_str + " " * (69 - len(cat_str)) + "│")
            
            print("└" + "─" * 34 + "┴" + "─" * 33 + "┘")
            print()
        
        # Best and worst performing tasks for each split
        for split in ['In_Domain', 'Out_of_Domain']:
            if results[split]['by_task']:
                task_scores = [(name, data['mean_score']) for name, data in results[split]['by_task'].items()]
                task_scores.sort(key=lambda x: x[1], reverse=True)
                
                print(f"{split} - Top 5 Tasks:")
                for name, score in task_scores[:5]:
                    short_name = name[:45] + "..." if len(name) > 48 else name
                    print(f"  {short_name}: {score:.4f}")
                
                print(f"{split} - Bottom 5 Tasks:")
                for name, score in task_scores[-5:]:
                    short_name = name[:45] + "..." if len(name) > 48 else name
                    print(f"  {short_name}: {score:.4f}")
                print()
        
        print("=" * 70)


# Convenience function for quick evaluation
def evaluate(
    videos_path: str,
    gt_base_path: str,
    output_path: str = './evaluation_results/',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run VBVR-Bench evaluation.
    
    Args:
        videos_path: Path to videos to evaluate
        gt_base_path: Path to ground truth data
        output_path: Path to save results
        **kwargs: Additional arguments passed to VBVRBench.evaluate()
        
    Returns:
        Evaluation results dictionary with In_Domain and Out_of_Domain scores
    """
    bench = VBVRBench(gt_base_path, output_path)
    return bench.evaluate(videos_path, **kwargs)
