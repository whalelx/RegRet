import pandas as pd
import os
import shutil
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datasets import load_dataset

image_output_dir = '/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/realworldqa/image'
output_json_path = './realworldqa.json'

ds = load_dataset("/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/realworldqa")

os.makedirs(image_output_dir, exist_ok=True)

sharegpt_format_data = []
for index, row in tqdm(enumerate(ds['test'])):
    # Extract image path and load the image
    image_path = f"{index}.jpg"
    image = row['image']
    
    # Save the image in the output directory
    image_output_path = os.path.join(image_output_dir, image_path)
    if index ==0:
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
    image.save(image_output_path)
    post_prompt = "Identify the regions that can help you answer the question, and then answer the question."

    # Construct the data for ShareGPT format
    sharegpt_item = {
        "messages":[
            {
                "role": "user",
                "content": "<image>\n" + row["question"] + post_prompt
            },
            {
                "role": "assistant",
                "content": "Now answer the question.\n"+f"{row['answer']}"
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
