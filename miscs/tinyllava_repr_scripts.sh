# install lmms_eval without building dependencies
cd lmms_eval;
pip install --no-deps -U -e .

# install TinyLLaVA without building dependencies
cd ..
git clone https://github.com/TinyLLaVA/TinyLLaVA_Factory
cd TinyLLaVA_Factory
pip install --no-deps -U -e .

# install all the requirements that require for reproduce llava results
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r tinyllava_repr_requirements.txt

# Run and reproduce tinyllava best results!
accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained=tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B,conv_mode=phi \
    --tasks vqav2,gqa,scienceqa_img,textvqa,mmvet,pope,mme,mmmu_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix tinyllava-phi2-siglip-3.1b \
    --output_path ./logs/