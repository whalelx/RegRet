import pandas as pd
import os
import shutil
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import csv
import argparse
import sys
import base64
from pathlib import Path
csv.field_size_limit(131072 * 10000000)  # 将限制增加到原来的10倍

image_output_dir = '/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/MMVP/image'
output_json_path = './mmvp.json'

headers = []
ds = []

prompt = "Please answer directly with only the letter of the correct option and nothing else. Identify the regions that can help you answer the question, and then answer the question."

with open("/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/MMVP/Questions.csv", 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=',')
    
    headers = next(reader)
    for row in reader:
        if row:  # 跳过空行
            ds.append(row)


sharegpt_format_data = []
for d in ds:
    index,  question,  choices ,ans = d[0],d[1],d[2],d[3]

    text = question + "\n" + choices + "\n" + prompt

    image_output_path = os.path.join(image_output_dir, f"{index}.jpg")

    sharegpt_item = {
        "messages":[
            {
                "role": "user",
                "content": "<image>\n" + text
            },
            {
                "role": "assistant",
                "content": ans
            },
        ],
        'images': [image_output_path]
    }
    sharegpt_format_data.append(sharegpt_item)

# Step 4: Save the ShareGPT format data as a JSON file
import json
with open(output_json_path, 'w') as f:
    json.dump(sharegpt_format_data, f, indent=4)

print(f"{output_json_path} data has been saved ")