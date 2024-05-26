source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh
conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval
export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH

accelerate launch --config_file /import/ml-sc-scratch5/etashg/scbx/deepspeed_zero3_2workers.yaml \
  --num_processes 1 -m lmms_eval --model llava \
  --model_args pretrained="liuhaotian/llava-v1.5-7b"   \
  --tasks xm3600_ja,xm3600_th,xm3600_ar,xm3600_hu  \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_v1.5_xm \
  --output_path ./logs/ 
