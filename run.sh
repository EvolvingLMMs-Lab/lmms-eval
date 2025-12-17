
  //VisualPuzzles
  
 python -m lmms_eval \
        --model bagel \
        --model_args pretrained=/scratch/models/BAGEL-7B-MoT,mode=understanding \
        --tasks VisualPuzzles_direct_categorized \
        --batch_size 1 \
        --limit 100 \
        --device cuda:0 \
        --output_path ./logs/bagel_visualpuzzles_direct_cat
        
python -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
        --tasks VisualPuzzles_direct_categorized  \
        --batch_size 1 \
        --limit 100 \
        --device cuda:0 \
        --output_path ./logs/qwen25_visualpuzzles_direct


  //MMSI-Bench
 python -m lmms_eval \
        --model bagel \
        --model_args pretrained=/scratch/models/BAGEL-7B-MoT,mode=understanding \
        --tasks mmsi_bench \
        --batch_size 1 \
        --limit 100 \
        --device cuda:0 \
        --output_path ./logs/bagel_visualpuzzles_cot



python -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
        --tasks mmsi_bench \
        --batch_size 1 \
        --limit 100 \
        --device cuda:0 \
        --output_path ./logs/qwen25_visualpuzzles_direct

// v=vi1