source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh
conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval
export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH

# accelerate launch --config_file /import/ml-sc-scratch5/etashg/scbx/deepspeed_zero3_2workers.yaml \
#   --num_processes 1 -m lmms_eval --model paligemma \
#   --model_args pretrained="google/paligemma-3b-mix-224" \
#   --tasks xm3600_ja,xm3600_th,xm3600_ar,xm3600_hu  \
#   --batch_size 16 \
#   --log_samples \
#   --log_samples_suffix paligemma-3b-mix-224_xm \
#   --output_path ./logs/ 


lmms_eval --model paligemma \
  --model_args pretrained="google/paligemma-3b-mix-224" \
  --tasks xm3600_ja_nl,xm3600_th_nl,xm3600_ar_nl,xm3600_hu_nl,xm3600_en_nl \
  --batch_size 16 \
  --log_samples \
  --log_samples_suffix paligemma-3b-mix-224_xm \
  --output_path ./logs/ 
