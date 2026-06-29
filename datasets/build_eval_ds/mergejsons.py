import os
import json
rootdir = "/data/spatialRGPT_eval/"
json_dir = os.path.join(rootdir, "jsons")
datalist=[]
post_prompt = "Identify the regions that can help you answer the question, and then answer the question."
for f in os.listdir(json_dir):
    if f.endswith(".json"):
        with open(os.path.join(json_dir, f), "r") as file:
            item = json.load(file)[0]
            item["image_path"] = os.path.join(rootdir,"images",item["image_path"])
            datalist.append(item)

# with open("../SpatialRGPT-Bench_v1.json", "w") as outfile:
#     json.dump(datalist, outfile, indent=4)

# with open("/data/dataset-spatial-reasoning/spatialrgpt/SpatialRGPT-Bench_v1.json", "r") as file:
#     data = json.load(file)
#     for item in data:
#         item["image_info"]["file_path"]= os.path.join("/data/spatialRGPT_eval/images/",item["id"],"0.png")
#     with open("/root/VisualPerceptionToken/datasets/SpatialRGPT-Bench_v1.json", "w") as outfile:
#         json.dump(data, outfile, indent=4)