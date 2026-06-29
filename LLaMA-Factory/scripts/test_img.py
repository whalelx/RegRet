from openai import OpenAI

api_key =  "0"
url =  "http://0.0.0.0:8000/v1"

# /data/spatialRGPT_qa/images/00000000/3c486192595fb49a/6_128,0,1024,557.png
systemprompt = "You are a helpful spatial reasoning assistant, good at analyzing fine-grained image features.In the image 'Region [i]' is the blue bounding box with the number i on the top left. Let's think step by step and start by identifying reference scales or object parts in the image, then answer the question.\n"

cotprompt2 = "Identify all the regions that can help you answer the question, tell me why it helps, and then answer the question."
# cotprompt2 = "Identify all the regions that can help you answer the question, and then answer the question."
# cotprompt2 = "Identify the region that can help you answer the question, and then answer the question."

image_urls = ["/data/dataset-spatial-reasoning/openimagev7/train/3d8edccaef77c1e7.jpg",
"/data/spatialRGPT_qa/images/00000000/3b31d5a6507ca16d/8.png",
"/data/spatialRGPT_qa/images/00000000/3c486192595fb49a/6.png",
"/data/spatialRGPT_qa/images/00000000/3d594dfc22414a49/4.png",
      "/data/dataset-spatial-reasoning/VSR/data/images/000000302514.jpg",
      "/data/dataset-spatial-reasoning/VSR/data/images/000000040934.jpg",
      "/data/spatialRGPT_qa/images/00000000/3a88f275daee0020/8.png",
        "/mnt/tidal-alsh01/usr/liangxun/data/dataset-spatial-reasoning/Qspatial/QSpatial_plus/images/018.jpeg",

]

questions = ["<image>\nWhat is the people that wears the glasses doing?\n",
"<image>\nWhat is the width of Region [0]?\n",
"<image>\nIf you are at Region [0], where will you find Region [1]?",
"<image>\nDoes Region [0] have a smaller size compared to Region [1]?",
# "<image>\nIs the Region [0] parallel to the Region [1]?",
"<image>\nIs the truck parallel to the motorcycle?",
"<image>\nIs the pizza near the sandwich?",
"<image>\nWhich is above, Region [0] or Region [1]",
"What is the minimum distance between the coffee grinder and the cubic rock in the image?"
]

idx=-1
image_url, question = image_urls[idx], questions[idx]

client = OpenAI(api_key=api_key,base_url=url)
messages = [
    # {
    #     "role": "system",
    #     "content": systemprompt
    # },
    {
        "role": "user", 
        "content": [
            {
                "type": "text", 
                "text": "Is the truck parallel to the motorcycle?\nthen answer the question."
                # "text":question  + cotprompt2
                # "text":systemprompt + question + cotprompt2
                # "text":systemprompt + question
            },
            {
                "type": "image_url", 
                "image_url": {"url": "/mnt/tidal-alsh01/usr/liangxun/data/visual-spatial-reasoning/images/000000302514.jpg"}
            },
        ]
    }
]

result = client.chat.completions.create(messages=messages, model="deepseek-v3", temperature=0.1)
print(result.choices[0].message)
