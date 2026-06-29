import json
from PIL import Image
import re
import cv2
import copy
import os
import json
import re
import tqdm
import numpy as np

QA_PER_FILE = 4

out_file_path = "../../LLaMA-Factory/data/osd-adunt.json"
root_dir = "/data/spatialRGPT_qa/"

Action_tokens = {
    "region_x": "<|x_0|>,<|x_1|>,<|x_2|>,<|x_3|>,<|x_4|>,<|x_5|>,<|x_6|>,<|x_7|>".split(","),
    "region_y": "<|y_0|>,<|y_1|>,<|y_2|>,<|y_3|>,<|y_4|>,<|y_5|>,<|y_6|>,<|y_7|>".split(","),
}

def from_region_tokens_to_region(region_tokens):
    match_tokens_x, match_tokens_y = region_tokens
    x_indices = [int(item.replace("<|x_","").replace("|>","")) for item in match_tokens_x]
    y_indices = [int(item.replace("<|y_","").replace("|>","")) for item in match_tokens_y]
    minx = min(x_indices)
    miny = min(y_indices)
    maxx = max(x_indices)
    maxy = max(y_indices)
    region = [(minx, miny), (maxx, maxy)]
    return region

def region_token_to_new_image_path(found_region_tokens, meta_item, num_cut=8):
    regions = from_region_tokens_to_region(found_region_tokens)
    max_x = max([x for x,y in regions])
    min_x = min([x for x,y in regions])
    max_y = max([y for x,y in regions])
    min_y = min([y for x,y in regions])

    max_x = min(max_x+1, num_cut-1)
    min_x = max(min_x-1, 0)
    max_y = min(max_y+1, num_cut-1)
    min_y = max(min_y-1, 0)
    image_path = meta_item["images"][0]
    image = Image.open(image_path)
    width, height = image.size
    st_w = 0
    st_h = 0

    grid_width = width / num_cut
    grid_height = height / num_cut

    new_x_min = st_w + (min_x ) * grid_width
    new_y_min = st_h + (min_y ) * grid_height
    new_x_max = st_w + (max_x + 1) * grid_width
    new_y_max = st_h + (max_y + 1) * grid_height

    new_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
    bbox_subfix = f"_{int(new_bbox[0])},{int(new_bbox[1])},{int(new_bbox[2])},{int(new_bbox[3])}"

    new_image_path = image_path + bbox_subfix
    return new_image_path

def generate_region_item(gt_item, ls_found_region_tokens, num_cut=8):

    ls_new_image_path = [region_token_to_new_image_path(found_region_tokens, gt_item, num_cut=num_cut) for found_region_tokens in ls_found_region_tokens]
    
    if len(ls_new_image_path) == 1:
        new_image_prompt = "<image>"
    else:
        new_image_prompt = ""
        for i in range(len(ls_new_image_path)):
            new_image_prompt += f"<image>\n"
    return ls_new_image_path, new_image_prompt


def check_region_tokens(text):

    pattern = re.compile(r'<\|region_token_start\|>(<\|[xy]_[01234567]\|>)+<\|region_token_end\|>')
    matches = pattern.finditer(text)

    found_tokens = []
    for match in matches:
        match_str = match.group()
        # match_str = match_str.replace("<|region_token_start|>","").replace("<|region_token_end|>","")
        match_tokens_x = [token for token in Action_tokens["region_x"] if token in match_str]
        match_tokens_y = [token for token in Action_tokens["region_y"] if token in match_str]
        found_tokens.append((match_tokens_x, match_tokens_y))

    if found_tokens:
        return found_tokens, True
    else:
        return None, False
    
def simplify_cot_format(text):
    pattern = re.compile(r'<\|region_token_start\|>(<\|[xy]_[01234567]\|>)+<\|region_token_end\|>')
    matches = pattern.finditer(text)
    found_tokens = []
    for i, match in enumerate(matches):
        match_str = match.group()
        found_tokens.append(match_str+"\n")
    return " ".join(found_tokens)


def convert_box_to_region(box, image_width , image_height, num_cut=None):
    num_cut = len(Action_tokens["region_x"]) if num_cut is None else num_cut


    grid_width = image_width / num_cut  # 80.0
    grid_height = image_height / num_cut # 60.0

    x1, y1, x2, y2 = box

    min_x_index = int(x1 // grid_width)     # 100 // 80 = 1
    min_y_index = int(y1 // grid_height)    # 50 // 60 = 0
    max_x_index = int((x2 - 1) // grid_width) # 299 // 80 = 3
    max_y_index = int((y2 - 1) // grid_height) # 199 // 60 = 3

    x_tokens = [f"<|x_{i}|>" for i in (min_x_index, max_x_index)] # ['<|x_1|>', '<|x_2|>', '<|x_3|>']
    y_tokens = [f"<|y_{i}|>" for i in (min_y_index, max_y_index)] # ['<|y_0|>', '<|y_1|>', '<|y_2|>', '<|y_3|>']

    found_region_tokens = "<|region_token_start|>" + x_tokens[0]+y_tokens[0]+x_tokens[1]+y_tokens[1] + "<|region_token_end|>\n"
    return found_region_tokens

import random
def run(chunk_name, pick=1):
    data_path = os.path.join(root_dir,"jsons/", chunk_name)
    datalist = []

    for file_name in os.listdir(data_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_path, file_name)

            with open(file_path, 'r') as f:
                data = json.load(f)

            for item in random.sample(data, QA_PER_FILE):
                new_item = {}
                new_item["images"] = [os.path.join(root_dir, f"images/{chunk_name}/{item['image_path']}")]
                regions = "".join([convert_box_to_region(box_list, item["image_width"] , item["image_height"]) for box_list in item["mask_list"]])
                ls_found_region_tokens, do_region = check_region_tokens(regions)
                ls_new_image_path, new_image_prompt = generate_region_item(new_item, ls_found_region_tokens , num_cut=8)
                new_item["images"] += ls_new_image_path
                
                # if system is not None:
                #     new_item["system"] = system

                question = item["question"].replace("<image>\n","")
                new_item["messages"] = [
                    {
                        "role": "user",
                        "content": "<image>\n" + question + "Identify the regions that can help you answer the question, and then answer the question."
                    },
                    {
                        "role": "assistant",
                        "content": regions
                    },
                    {
                        "role": "user",
                        "content": new_image_prompt
                    },
                    {
                        "role": "assistant",
                        "content": "Now answer the question.\n" + item["answer"]
                    }
                ]
                datalist.append(new_item)
    return datalist
        


import concurrent.futures

if __name__ == "__main__":
    chunk_list = [str(chunk_idx).zfill(8) for chunk_idx in range(50)]
    max_workers = 50  # Consider adjusting this based on your system's capabilities
    
    aggregated_datalist = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run, chunk, 2) for chunk in chunk_list]
        print(f"Submitted {len(futures)} tasks to process pool.")
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
            try:
                chunk_result = future.result()
                if chunk_result is not None:
                    aggregated_datalist.extend(chunk_result)
            except Exception as e:
                print(f"An error occurred while processing a chunk: {e}")

    print(f"All chunks processed. Total items collected: {len(aggregated_datalist)}")

    try:
        with open(out_file_path, 'w') as f:
            json.dump(aggregated_datalist, f, indent=2)
        print(f"Successfully aggregated data and wrote to {out_file_path}")
    except IOError as e:
        print(f"Error writing to file {out_file_path}: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}. Ensure all items in aggregated_datalist are JSON serializable.")
