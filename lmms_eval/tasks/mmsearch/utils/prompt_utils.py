import os
import re
from lmms_eval.tasks.mmsearch.utils.image_utils import slim_image_and_save, crop_and_split

DEFAULT_IMAGE_TOKEN = '<image>'


def get_website_information(result_brief):
    '''
    result_brief: [{'title', 'text','screenshot_path'}]
    '''
    website_information, input_image_list = [], []
    for idx, inst in enumerate(result_brief):
        template = f"Website {idx+1} Title: {inst['title']};\nWebsite {idx+1} snippet: {inst['snippet']};\nWebsite {idx+1} Screenshot: {DEFAULT_IMAGE_TOKEN}"
        website_information.append(template)
        input_image_list.append(inst['screenshot_path'])

    return '\n\n'.join(website_information), input_image_list

def get_rerank_incontext_example(rerank_num):
    l = [f"<Website {i}>" for i in range(rerank_num)]
    return ','.join(l)

def get_full_website_information(result_full, image_dir='', fullpage_split_dict=None, save_slim_dir=None):
    '''
    result_full: [{'title', 'snippet', 'content','screenshot_path'}]
    '''
    if save_slim_dir is None:
        save_slim_dir = image_dir

    input_image_list = []
    inst = result_full[0] # assert only 1 fullpage content

    template = f"Website Title: {inst['title']};\n Website Snippet: {inst['snippet']};\n"
    
    # add content
    template += f"Website Content: {inst['content']};\n"
    
    ## slim image to be tense
    if 'screenshot_fullpage_path' in inst:
        split_list = inst['screenshot_fullpage_path'].split('/')[-1].split('.')
        save_name = '.'.join(split_list[:-1]) + f"_slim.{split_list[-1]}"
        save_path = os.path.join(save_slim_dir, save_name)

        slim_image_and_save(
            image_path=os.path.join(image_dir, inst['screenshot_fullpage_path']),
            save_path=save_path
        )
        save_slice_path = os.path.join(save_slim_dir, 'slices')
        os.makedirs(save_slice_path, exist_ok=True)
    elif 'slimmed_website_fullpage_screenshot' in inst: # the screenshot is already slimmed
        save_path = inst['slimmed_website_fullpage_screenshot']
        save_slice_path = None # do not save the slices
    else:
        raise ValueError('seems that the inst variable does not contain relevant key')

    # here, we split the fullpage to maximum 10 images, each with 512 height (the width depends on the website itself)
    screenshot_fullpage_split_list = crop_and_split(
        fullpage_path=save_path,
        fullpage_split_dict=fullpage_split_dict,
        save_slice_path=save_slice_path
    )
    template += f"Website Screenshot: {DEFAULT_IMAGE_TOKEN*len(screenshot_fullpage_split_list)};\n"
    input_image_list.extend(screenshot_fullpage_split_list)

    website_information = template
        
    return website_information, input_image_list

def postprocess_rerank(rerank, rerank_num):
    pattern = r'<Website (\d+)>'
    matches = re.findall(pattern, rerank)
    output_index = [int(x)-1 for x in matches]
    if len(output_index) > rerank_num:
        print(f'More index than rerank number: {rerank}')
        output_index = output_index[:rerank_num]
        valid = False
    elif len(output_index) < rerank_num:
        print(f'Less index than rerank number: {rerank}')
        if len(output_index) == 0:
            print('No valid output for rereank')
            output_index = [i for i in range(rerank_num)]
        valid = False
    elif not all([[x < 0 for x in output_index]]):
        print(f'Some index is less than 1: {rerank}')
        output_index = [i for i in range(rerank_num)]
        valid = False
    else:
        valid = True
    
    return output_index, valid