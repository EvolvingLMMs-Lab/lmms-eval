#!/usr/bin/env python3
"""
Direct test of UniWorld model loading and generation
Compare our implementation with original UniWorld CLI
"""

import sys
import os
import torch
from PIL import Image

# Add UniWorld path
sys.path.insert(0, '/home/aiscuser/lmms-eval/UniWorld/UniWorld-V1')

from transformers import AutoProcessor, set_seed
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

set_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "LanguageBind/UniWorld-V1"
    
    print("=" * 60)
    print("Loading model...")
    print("=" * 60)
    
    # Load exactly like original CLI
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Original uses this!
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        model_path, 
        min_pixels=448*448, 
        max_pixels=448*448
    )
    
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    
    # Test with a simple text prompt (no image)
    print("\n" + "=" * 60)
    print("Test 1: Text-only generation")
    print("=" * 60)
    
    conversation = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2? Answer with a single number."}]}]
    chat_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # Drop system message like original CLI
    chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])
    
    print(f"Chat text:\n{chat_text}")
    
    inputs = processor(
        text=[chat_text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    print(f"\nInput IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs (last 20): {inputs.input_ids[0, -20:].tolist()}")
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    
    print(f"\nGenerated IDs shape: {generated_ids.shape}")
    print(f"Generated IDs (last 30): {generated_ids[0, -30:].tolist()}")
    
    # Decode only new tokens
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    print(f"Trimmed tokens: {trimmed[0].tolist()}")
    
    reply = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"\n>>> Generated reply: '{reply}'")
    
    # Test 2: With a sample image from chartqa
    print("\n" + "=" * 60)
    print("Test 2: Image + text generation")
    print("=" * 60)
    
    # Use a local test image or download one
    test_image_path = "/blob/lmms-eval-dataset/chartqa100_images/4.png"  
    if os.path.exists(test_image_path):
        conversation2 = [
            {"role": "user", "content": [
                {"type": "image", "image": test_image_path, "min_pixels": 448*448, "max_pixels": 448*448},
                {"type": "text", "text": "What is the title of this chart? Answer briefly."}
            ]}
        ]
        
        chat_text2 = processor.apply_chat_template(conversation2, tokenize=False, add_generation_prompt=True)
        chat_text2 = '<|im_end|>\n'.join(chat_text2.split('<|im_end|>\n')[1:])
        
        image_inputs, video_inputs = process_vision_info(conversation2)
        print(f"Image inputs: {len(image_inputs) if image_inputs else 0}")
        
        inputs2 = processor(
            text=[chat_text2],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        print(f"Input IDs shape: {inputs2.input_ids.shape}")
        print(f"Has pixel_values: {hasattr(inputs2, 'pixel_values') and inputs2.pixel_values is not None}")
        if hasattr(inputs2, 'pixel_values') and inputs2.pixel_values is not None:
            print(f"Pixel values shape: {inputs2.pixel_values.shape}")
        
        with torch.inference_mode():
            generated_ids2 = model.generate(**inputs2, max_new_tokens=32, do_sample=False)
        
        trimmed2 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs2.input_ids, generated_ids2)]
        print(f"Trimmed tokens: {trimmed2[0].tolist()}")
        
        reply2 = processor.batch_decode(trimmed2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"\n>>> Generated reply: '{reply2}'")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Skipping image test")

if __name__ == "__main__":
    test_model()
