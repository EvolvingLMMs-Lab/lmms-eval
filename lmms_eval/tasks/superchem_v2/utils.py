import os
import hashlib
import re
import io
from PIL import Image

IMAGE_SAVE_DIR = "./superchem_en_images"

if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
PROMPT_TEMPLATE = r"""Question:

{question}

{options}

Select the best answer to the above multiple-choice question based on the image. Respond with only the letter of the correct option."""

def get_options_string(doc, lang="en"):
    options_dict = doc.get(f'options_{lang}', {})
    options_str = ''
    for key in sorted(options_dict.keys()):
        if options_dict.get(key) is not None:
            options_str += f"{key}: {options_dict[key]}\n"
        else:
            break
    return options_str

def doc_to_visual(doc, kwargs=None):
    lang = "en"
    question = doc.get(f'question_{lang}', '')
    options = get_options_string(doc, lang)
    prompt = PROMPT_TEMPLATE.replace('{question}', question).replace('{options}', options)
    
    question_images = doc.get('question_images') or {}
    options_images = doc.get('options_images') or {}
    images_dict = {**question_images, **options_images}

    visuals = []
    tag_pattern = r'<MultiModal>(.*?)</MultiModal>'
    link_pattern = r'!?\s*\[(.+)\]\s*\(([^)]+)\)'
    
    for tag_match in re.finditer(tag_pattern, prompt, re.IGNORECASE | re.DOTALL):
        tag_content = tag_match.group(1)
        link_match = re.match(link_pattern, tag_content, flags=re.MULTILINE | re.DOTALL)
        
        if link_match:
            image_url = link_match.group(2)
            image_bytes = images_dict.get(image_url)
            
            if image_bytes is None:
                raise FileNotFoundError(f"Image file not found: {image_url}")
            
            if image_bytes.startswith(b'\xff\xd8\xff'):
                mimetype = 'jpeg'
            elif image_bytes.startswith(b'\x89PNG'):
                mimetype = 'png'
            else:
                raise ValueError(f"Unsupported image format for {image_url}")

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            visuals.append(img)
        else:
            raise ValueError(f"No link found in multimodal tag: {tag_content}")
            
    return visuals

def doc_to_text(doc, kwargs=None):
    """
    文本占位符替换逻辑
    """
    lang = "en"
    question = doc.get(f'question_{lang}', '')
    options = get_options_string(doc, lang)
    prompt = PROMPT_TEMPLATE.replace('{question}', question).replace('{options}', options)
    
    tag_pattern = r'<MultiModal>(.*?)</MultiModal>'

    text = re.sub(tag_pattern, "<image>", prompt, flags=re.IGNORECASE | re.DOTALL)
    return text

def doc_to_target(doc, kwargs=None):
    return {
        "ref": doc.get('answer_en', []),
        "type": doc.get('question_type', '')
    }

def mcp_metric(results, data):
    total_score = 0
    for response, target in zip(results, data):
        pattern = r'\\boxed\{(.+?)\}'
        matches = list(re.finditer(pattern, response, re.DOTALL))
        
        if not matches: continue
        
        answer = matches[-1].group(1).strip()
        for char in ['\\', '{', '}', 'text', 'math', 'bf', 'rm']:
            answer = answer.replace(char, '')
        answer = answer.strip()

        ref_answer = target["ref"]
        q_type = target["type"]
        
        if q_type == 'multiple_choice':
            if len(answer) == len(ref_answer) and all(a.upper() in ref_answer for a in answer):
                total_score += 1
        elif q_type == 'fill_blank':
            if ref_answer and answer.lower() == ref_answer[0].lower():
                total_score += 1
                
    return total_score / len(results) if results else 0

def save_image_from_bytes(img_bytes, sample_id, img_name):
    """
    保存图片到指定目录。
    sample_id: 样本的唯一ID (uuid)
    img_name: 图片在题目中的原始名称 (如 image_0.png)
    """
    try:
        # 清洗文件名，防止路径遍历漏洞
        safe_sample_id = str(sample_id).replace("/", "_").replace("\\", "_")
        safe_img_name = str(img_name).replace("/", "_").replace("\\", "_")
        
        # 构造完整路径
        filename = f"{safe_sample_id}_{safe_img_name}"
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += ".png"
            
        file_path = os.path.join(IMAGE_SAVE_DIR, filename)

        # 如果文件已存在，直接返回绝对路径，跳过 IO
        if os.path.exists(file_path):
            return os.path.abspath(file_path)

        # 执行保存
        if isinstance(img_bytes, bytes):
            with open(file_path, "wb") as f:
                f.write(img_bytes)
        elif isinstance(img_bytes, Image.Image):
            img_bytes.save(file_path)
        
        return os.path.abspath(file_path)
    except Exception as e:
        return f"Error saving image: {str(e)}"

def process_results(doc, results):
    """
    处理单个样本的预测结果
    """
    model_output = str(results[0]) if results else ""
    
    # --- 答案解析逻辑 ---
    pattern = r'\\boxed\{(.+?)\}'
    matches = list(re.finditer(pattern, model_output, re.DOTALL))
    
    # 如果没找到 boxed，回退到使用全部输出
    content = matches[-1].group(1).strip() if matches else model_output.strip()
    
    # 清洗特殊字符
    for char in ['\\', '{', '}', 'text', 'math', 'bf', 'rm']:
        content = content.replace(char, '')
    for char in ["[", "]", "'", '"', ","]:
        content = content.replace(char, '')
    
    # 提取最终的字母/数字答案
    parsed_answer = "".join(re.findall(r'[A-Za-z0-9]', content))

    # --- 评分逻辑 ---
    score = 0
    ref_answer = doc.get('answer_en', [])
    q_type = doc.get('question_type', 'multiple_choice')
    
    if q_type == 'multiple_choice':
        if len(parsed_answer) == len(ref_answer) and all(l.upper() in ref_answer for l in parsed_answer):
            score = 1
    elif q_type == 'fill_blank':
        if ref_answer and parsed_answer.lower() == str(ref_answer[0]).lower():
            score = 1

    # --- 图片保存逻辑 ---
    sample_id = doc.get("uuid") or doc.get("id") or hashlib.md5(model_output[:50].encode()).hexdigest()
    
    image_paths = []
    question_images = doc.get('question_images') or {}
    options_images = doc.get('options_images') or {}
    all_imgs = {**question_images, **options_images}
    
    for img_name, img_bytes in all_imgs.items():
        saved_path = save_image_from_bytes(img_bytes, sample_id, img_name)
        image_paths.append(saved_path)

    # --- 返回结果 ---
    return {
        "mcp_accuracy": score,
        "groundtruth": ref_answer,
        "raw_output": model_output,
        "question": doc_to_text(doc),
        "parsed_answer": parsed_answer,
        "image_path": image_paths
    }