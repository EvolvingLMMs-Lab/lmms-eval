source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh
conda activate /import/ml-sc-scratch5/etashg/miniconda3/envs/mplug_owl2
export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/import/ml-sc-scratch5/etashg/mPLUG-DocOwl/DocOwl1.5

lmms_eval --model docowl \
  --tasks hu_docvqa \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix docowl_hu \
  --output_path ./logs/ 
