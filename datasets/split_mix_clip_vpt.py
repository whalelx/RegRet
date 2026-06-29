import json, os
from tqdm import tqdm
root= "../LLaMA-Factory/data/"
file_path = os.path.join(root, "MixVRT_CLIP_Full.json")

ds = ["GQA/", "COCO",  "OpenImage", "VSR", "TextVQA", "OCRVQA", "DocVQA"]
try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    dataset_spatial = []
    dataset_textocr = []
    dataset_llava = []
    dataset_spatial_region_cnt =0
    dataset_text_region_cnt =0
    
    for item in tqdm(data):
        if len(item["clip_images"]) > 0:    
            continue
        file_path = item["images"][0]

        del item["clip_images"]
        del item["detection_images"]
        del item["seg_images"]

        flag = 1 if len(item["images"]) > 1 else 0

        for id, name in enumerate(ds):
            if name in file_path:

                if id<=3:
                    if flag == 0:
                        dataset_llava.append(item)
                    else:
                        dataset_spatial.append(item)
                        dataset_spatial_region_cnt += flag
                    break
                else:
                    dataset_text_region_cnt += flag
                    dataset_textocr.append(item)
                break
        else:
            print(f"Error: {file_path} not in dataset list")
            exit()

        
    with open(os.path.join(root, "mix-vpt-spatial.json"), 'w') as f:
        f.write(json.dumps(dataset_spatial, indent=4))
    with open(os.path.join(root, "mix-vpt-text.json"), 'w') as f:
        f.write(json.dumps(dataset_textocr, indent=4))
    with open(os.path.join(root, "mix-vpt-llava.json"), 'w') as f:
        f.write(json.dumps(dataset_llava, indent=4))

    print(f"spatial length: {len(dataset_spatial)}, text length: {len(dataset_textocr)}, llava length: {len(dataset_llava)}")
    print(f"spatial region cnt: {dataset_spatial_region_cnt}, text region cnt: {dataset_text_region_cnt}")

    print(f"Successfully added system prompt and updated {file_path}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
