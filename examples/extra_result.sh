#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_results_directory>"
    echo "Example: $0 /mnt/cpfs/yangyicun/eval_result/model__Qwen3-VL-8B-Instruct"
    exit 1
fi

RESULTS_DIR=$1

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory $RESULTS_DIR does not exist."
    exit 1
fi

echo "=========================================================="
echo "Extracting evaluation results from: $RESULTS_DIR"
echo "=========================================================="
echo ""

# Process all json files through a single python script to handle grouping by date
# Using '|| true' to prevent set -e from exiting if grep finds nothing
find "$RESULTS_DIR" -name "*.json" | grep -v "samples" > /tmp/json_files.txt || true

if [ ! -s /tmp/json_files.txt ]; then
    echo "No valid JSON result files found."
    rm -f /tmp/json_files.txt
    exit 0
fi

python3 -c "
import sys, json, os

with open('/tmp/json_files.txt', 'r') as f:
    files = f.read().splitlines()

results_by_date = {}

for json_file in files:
    if not json_file.strip():
        continue
    filename = os.path.basename(json_file)
    # Extract date prefix (e.g., 20260323_224426 from 20260323_224426_results.json)
    date_prefix = filename.split('_results')[0] if '_results' in filename else 'unknown_date'
    
    if date_prefix not in results_by_date:
        results_by_date[date_prefix] = []
        
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'results' in data:
            results_by_date[date_prefix].append((json_file, data['results']))
    except Exception as e:
        print(f'Error parsing JSON {json_file}: {e}')

# Sort dates to print them chronologically
for date_prefix in sorted(results_by_date.keys()):
    print(f'----------------------------------------------------------')
    print(f'Experiment Timestamp: {date_prefix}')
    print(f'----------------------------------------------------------')
    print(f'{\"Task\":<25} | {\"Metric\":<25} | {\"Value\":<10}')
    print('-'*65)
    
    for json_file, results in results_by_date[date_prefix]:
        for task, metrics in results.items():
            for metric_name, value in metrics.items():
                # Skip 'alias' and empty keys
                if metric_name == 'alias' or metric_name.strip() == '':
                    continue
                
                # Exclude standard error (stderr) and complex nested data (lists/dicts)
                if 'stderr' in metric_name or isinstance(value, (list, dict)):
                    continue
                    
                # Clean up metric names like 'mmmu_acc,none' -> 'mmmu_acc'
                clean_metric_name = metric_name.split(',')[0]
                
                if isinstance(value, float):
                    print(f'{task:<25} | {clean_metric_name:<25} | {value:.4f}')
                else:
                    print(f'{task:<25} | {clean_metric_name:<25} | {value}')
    print()
"

rm -f /tmp/json_files.txt

echo "=========================================================="
echo "Extraction Complete."
echo "=========================================================="
