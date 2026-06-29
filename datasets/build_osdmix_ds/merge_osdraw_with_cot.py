import json, os, re
from tqdm import tqdm
out_file_path = "../../LLaMA-Factory/data/osd-adunt.json"

with open(out_file_path, "r") as f:
    data0 = json.load(f)

with open("./osd-vpt-cot.json", "r") as f:
    data1 = json.load(f)

key2d = {}
for d in tqdm(data1):
    key = (d["images"][0], d["messages"][0]["content"])
    key2d[key] = d

dlist = []
i = 0
for item in tqdm(data0):
    key = (item["images"][0], item["messages"][0]["content"])
    if key in key2d:
        pass
    else:
        i+=1
        dlist.append(item)
print(len(dlist),i)

with open("../../LLaMA-Factory/data/mix-vpt-llava.json", "r") as f:
    data3 = json.load(f)[:200000]

with open("../../LLaMA-Factory/data/mix-vpt-spatial.json", "r") as f:
    data4 = json.load(f)

dlist.extend(data3)
dlist.extend(data1)
dlist.extend(data4)
import random
random.shuffle(dlist)

with open("../../LLaMA-Factory/data/xyz.json", "w") as f:
    json.dump(dlist, f, indent=2)
