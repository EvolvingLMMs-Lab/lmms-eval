# User Guide
This document details the running examples for different models in `lmms_eval`. We include commandas on how to prepare environments for different model and some commands to run these models

## Environmental Variables

Before running experiments and evaluations, we recommend you to export following environment variables to your environment. Some are necessary for certain tasks to run.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Some common environment issue
Sometimes you might encounter some common issues for example error related to `httpx` or `protobuf`. To solve these issues, you can first try

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

# Image Model

### LLaVA
First, you will need to clone repo of `lmms_eval` and repo of [`llava`](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference)

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/LLaVA-NeXT;
python3 -m pip install -e ".[train]";


TASK=$1
CKPT_PATH=$2
CONV_TEMPLATE=$3
MODEL_NAME=$4
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

#mmbench_en_dev,mathvista_testmini,llava_in_the_wild,mmvet
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
```
If you are trying to use large LLaVA models such as LLaVA-NeXT-Qwen1.5-72B, you can try adding `device_map=auto` in model_args and change `num_processes` to 1.

### IDEFICS2

You won't need to clone any other repos to run idefics. Making sure your transformers version supports idefics2 would be enough

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model idefics2 \
    --model_args pretrained=HuggingFaceM4/idefics2-8b \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### InternVL2

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;


python3 -m pip install flash-attn --no-build-isolation;
python3 -m pip install torchvision einops timm sentencepiece;


TASK=$1
CKPT_PATH=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12380 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```


### InternVL-1.5
First you need to fork [`InternVL`](https://github.com/OpenGVLab/InternVL)

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/InternVL/internvl_chat
python3 -m pip install -e .;

python3 -m pip install flash-attn==2.3.6 --no-build-isolation;


TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model internvl \
    --model_args pretrained="OpenGVLab/InternVL-Chat-V1-5"\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

```

### Xcomposer-4KHD and Xcomposer-2d5

Both of these two models does not require external repo

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;


python3 -m pip install flash-attn --no-build-isolation;
python3 -m pip install torchvision einops timm sentencepiece;

TASK=$1
MODALITY=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

# For Xcomposer2d5
accelerate launch --num_processes 8 --main_process_port 10000 -m lmms_eval \
    --model xcomposer2d5 \
    --model_args pretrained="internlm/internlm-xcomposer2d5-7b",device="cuda",modality=$MODALITY\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

# For Xcomposer-4kHD
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model xcomposer2_4khd \
    --model_args pretrained="internlm/internlm-xcomposer2-4khd-7b" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```

### InstructBLIP

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;

CKPT_PATH=$1
TASK=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model instructblip \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix instructblip \
    --output_path ./logs/ 

```

### SRT API MODEL
To enable faster testing speed for larger llava model, you can use this srt api model to enable testing through sglang.
You will need to first glone sglang from "https://github.com/sgl-project/sglang". Current version is tested on the commit #1222 of sglang

Here are the scripts if you want to test the result in one script.
```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/sglang;
python3 -m pip install -e "python[all]";


python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/


CKPT_PATH=$1
TASK=$2
MODALITY=$3
TP_SIZE=$4
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

python3 -m lmms_eval \
    --model srt_api \
    --model_args modality=$MODALITY,model_version=$CKPT_PATH,tp=$TP_SIZE,host=127.0.0.1,port=30000,timeout=600 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```

You can use the script in `sglang` under `test` folder to kill all sglang service

# API Model

### GPT

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

export OPENAI_API_KEY="<YOUR_API_KEY>"

TASK=$1
MODEL_VERSION=$2
MODALITIES=$3
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 30000 -m lmms_eval \
    --model gpt4v \
    --model_args model_version=$MODEL_VERSION,modality=$MODALITIES\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```


### Claude

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

export ANTHROPIC_API_KEY="<YOUR_API_KEY>"

TASK=$1
MODEL_VERSION=$2
MODALITIES=$3
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model claude \
    --model_args model_version=$MODEL_VERSION\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
```


# Video Model

### LLaVA-VID

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/LLaVA-NeXT;
python3 -m pip install -e ".[train]";

python3 -m pip install flash-attn --no-build-isolation;

python3 -m pip install av;


TASK=$1
CKPT_PATH=$2
CONV_TEMPLATE=$3
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llavavid \
    --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,video_decode_backend=decord,max_frames_num=32 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```


### LLaMA-VID

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

# Notice that you should not leave the folder of LLaMA-VID when calling lmms-eval
# Because they left their processor's config inside the repo
cd /path/to/LLaMA-VID;
python3 -m pip install -e .

python3 -m pip install av sentencepiece;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llama_vid \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
```

### Video-LLaVA

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;
python3 -m pip install av sentencepiece;


TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model video_llava \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```


### MPlug-Owl
Notice that this model will takes long time to load, please be patient :)

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

# It has to use an old transformers version to run
python3 -m pip install av sentencepiece protobuf==3.20 transformers==4.28.1 einops;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model mplug_owl_video \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

```


### Video-ChatGPT

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install sentencepiece av;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model video_chatgpt \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### MovieChat

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/rese1f/MovieChat.git
mv /path/to/MovieChat/MovieChat /path/to/lmms-eval/lmms_eval/models

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model moviechat \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### LLaVA-OneVision-MovieChat

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/rese1f/MovieChat.git
mv /path/to/MovieChat/MovieChat_OneVision/llava /path/to/lmms-eval/lmms_eval/models

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision_moviechat \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### AuroraCap

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/rese1f/aurora.git
mv /path/to/aurora/src/xtuner/xtuner /path/to/lmms-eval/lmms_eval/models/xtuner-aurora

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model auroracap \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```


### SliME

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/yfzhang114/SliME.git

cd SliME
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd ..

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model slime \
    --tasks $TASK \
    --model_args pretrained="yifanzhang114/SliME-Llama3-8B,conv_template=llama3,model_name=slime" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```
