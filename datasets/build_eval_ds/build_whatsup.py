import pandas as pd
import os
import shutil
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import json

dataset_path = '/mnt/tidal-alsh01/usr/liangxun/data/whatsup/whatsup_vlms/test.json'
output_json_path = './whatsup.json'

with open(dataset_path, 'r') as f:
    data = json.load(f)

sharegpt_format_data = []
for row in tqdm(data):
    # Extract image path and load the image
    image_path = os.path.join(os.path.dirname(dataset_path), row['filename'])
    correctidx = row["gold_index"]
    question = f"""
Which one of the following description is correct?
[{row["caption_options"][0]} ,{row["caption_options"][1]}, {row["caption_options"][2]},{row["caption_options"][3]}]. 
Output the correct index among 0,1,2,3.
"""

    post_prompt = "Identify the regions that can help you answer the question, and then answer the question."

    # Construct the data for ShareGPT format
    sharegpt_item = {
        "messages":[
            {
                "role": "user",
                "content": "<image>\n" + question + post_prompt
            },
            {
                "role": "assistant",
                "content": "Now answer the question.\n"+f"{correctidx}, {row['caption_options'][correctidx]}"
            },
        ],
        'images': [image_path]
    }
    sharegpt_format_data.append(sharegpt_item)

# Step 4: Save the ShareGPT format data as a JSON file
with open(output_json_path, 'w') as f:
    json.dump(sharegpt_format_data, f, indent=4)

print(f"{output_json_path} data has been saved ")
