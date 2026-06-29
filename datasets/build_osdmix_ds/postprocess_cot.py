import json
import re
import os,random

def extract_cot_groups_from_files(file_paths):
    for file_path in file_paths:
        print(f"\nProcessing file: {file_path}") # Corrected: \n for newline
        if not os.path.exists(file_path):
            print(f"  Error: File not found at {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            cot_value = item.get("cot")

            if cot_value != "" and isinstance(cot_value, str):
                match = re.search(r"\[COT: (.*?)\]\n", cot_value) 
                if match:
                    extracted_group = match.group(1)
                    answer = item["messages"][-1]["content"]
                    answer = answer.split("\n")
                    if len(answer) > 1:
                        answer = answer[0] + extracted_group + "Therefore, the answer is:" + answer[1]
                    else:
                        answer = extracted_group + "Therefore, the answer is:" + answer[0]
                    item["messages"][-1]["content"] = answer
                    item.pop("cot")
        file_path_ = file_path.split(".json")[0]
        with open(file_path_+"-processcot.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
                    

if __name__ == "__main__":

    root_dir = "../../LLaMA-Factory/data/"
    json_files_to_process = [
        "mix-vpt-spatial-cot.json",
        "osd-vpt-cot.json",
    ]
    extract_cot_groups_from_files(json_files_to_process)
    
    # input_files = [
    #     "mix-vpt-spatial-cot-processcot.json",
    #     "osd-vpt-cot-processcot.json",
    #     'mix-vpt-llava.json'
    # ]
    all_sampled_data = []

    with open(os.path.join(root_dir, 'mix-vpt-llava.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_sampled_data.extend(data)

    with open(os.path.join(root_dir, 'mix-vpt-spatial.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_sampled_data.extend(data[82005:])

    with open("mix-vpt-spatial-cot-processcot.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_sampled_data.extend(data)

    with open("osd-vpt-cot-processcot.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_sampled_data.extend(data)

    with open(os.path.join(root_dir, 'Mix_OSD_VPT_Cot450k.json'), 'w', encoding='utf-8') as f:
        random.shuffle(all_sampled_data)
        json.dump(all_sampled_data, f, indent=4, ensure_ascii=False)

    os.remove("mix-vpt-spatial-cot-processcot.json")
    os.remove("osd-vpt-cot-processcot.json")

    print("All files processed and saved to Mix_OSD_VPT_Cot450k.json")