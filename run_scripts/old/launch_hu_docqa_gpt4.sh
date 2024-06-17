source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh
conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mm_eval
export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH

export API_TYPE="openai"


lmms_eval --model gpt4v \
  --tasks hu_docvqa \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix gpt4v_v1.5_hu \
  --output_path ./logs/ 
