import pandas as pd
import os
import shutil
from PIL import Image
from io import BytesIO
from tqdm import tqdm

image_output_dir = '/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/blink/Spatial_Relation/images'
dataset_path = '/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/blink/Spatial_Relation/val-00000-of-00001.parquet'
output_json_path = './blink.json'

df = pd.read_parquet(dataset_path)
os.makedirs(image_output_dir, exist_ok=True)

sharegpt_format_data = []
for index, row in tqdm(df.iterrows()):
    # Extract image path and load the image
    image_path = row['idx']+".jpg"
    image = row['image_1']["bytes"]
    
    # Save the image in the output directory
    image_output_path = os.path.join(image_output_dir, image_path)
    if index ==0:
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
    image = Image.open(BytesIO(image))
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
                "content": f'{"yes" if row["answer"]=="(A)" else "No" }'
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
