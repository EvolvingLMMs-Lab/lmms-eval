# # STEP # 1: Install lmms-eval
# pip install lmms-eval

# # STEP # 2: Install perception_models (Details at https://github.com/facebookresearch/perception_models)
# git clone https://github.com/facebookresearch/perception_models.git
# cd perception_models

# conda create --name perception_models python=3.12
# conda activate perception_models

# # Install PyTorch
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

# # We use torchcodec for decoding videos into PyTorch tensors
# conda install ffmpeg -c conda-forge
# pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124

# pip install -e .


# Use facebook/Perception-LM-1B for 1B parameters model and facebook/Perception-LM-8B for 8B parameters model.
CHECKPOINTS_PATH=facebook/Perception-LM-3B

# Define the tasks you want to evaluate PLM on. We support all the tasks present in lmms-eval, however have tested the following tasks with our models.
ALL_TASKS=(
    "docvqa" "chartqa" "textvqa" "infovqa" "ai2d_no_mask" "ok_vqa" "vizwiz_vqa" "mme"
    "realworldqa" "pope" "mmmu" "ocrbench" "coco_karpathy_val" "nocaps" "vqav2_val"
    "mvbench" "videomme" "vatex_test" "egoschema" "egoschema_subset" "mlvu_dev"
    "tempcompass_multi_choice" "perceptiontest_val_mc" "perceptiontest_test_mc"
)

# We select one image and one video task as an example.
SELECTED_TASK="textvqa,videomme"

# After specifying the task/tasks to evaluate, run the following command to start the evaluation.
accelerate launch --num_processes=8 \
-m lmms_eval \
--model plm \
--model_args pretrained=$CHECKPOINTS_PATH,max_tokens=11264 \
--tasks $SELECTED_TASK \
--batch_size 1 \
--log_samples \
--log_samples_suffix plm \
--output_path plm_reproduce
