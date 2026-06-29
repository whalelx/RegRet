
from openai import OpenAI 
import os

client = OpenAI(
    api_key="",
    base_url=""
)

"""
example1: 流式调用LLM服务
"""
"""
completion = client.chat.completions.create(
    model="deepseek-v3-0324", #在Body中指明要访问的模型名，即“model”（见2.1表格1）
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
    stream=True,
    max_tokens=64,
    temperature=0.9
    )
for chunk in completion:
    print(chunk.model_dump_json())
"""

"""
example2: 非流式调用LLM服务
"""

# qwen3-235b-a22b
# qwen3-30b-a3b
# qwen3-32b
# qwen3-14b,
# qwen3-1.7b,
# deepseek-r1-distill-qwen-32b,
# qwen2.5-coder-32b-instruct,
# deepseek-r1-distill-llama-70b,
# qwen2.5-32b-instruct,
# deepseek-v3,
# deepseek-r1,
# qwen2.5-72b-instruct-func,l
# lama-3.3-70b-instruct,
# qwen3-4b,
# deepseek-r1-distill-qwen-1.5b,
# qwen2.5-72b-instruct,
# qwen2.5-14b-instruct,
# llama-4-scout-17b-16e-instruct,
# qwen2.5-coder-7b-instruct,
# deepseek-coder-v2-lite-instruct,
# deepseek-coder-33b-instruct,
# qwen2.5-72b-instruct-128k,
# deepseek-r1-distill-qwen-7b,
# qwen3-8b,qwen2.5-7b-instruct,deepseek-r1-32k,qwen2.5-72b-funtion-tool,,
# deepseek-v3-0324
# ,qwen3-0.

# qwq-32b
# qwen2.5-vl-32b-instruct
# qwen2.5-vl-72b-instruct
# qwen2.5-vl-7b-instruct
# xdg-vl-72b
image_url = "/data/spatialRGPT_qa/images/00000015/5ddaf30d2e20f64c/5.png"
      
image_url = "/data/openimagesv7/train/07d7abee3dd1c9d3.jpg"
image_url = "/data/openimagesv7/train/1ef3803bfcd64e54.jpg"
import base64

with open(image_url, "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode('utf-8')

template = 'There is a question "{:s}" and a correct answer "{:s}" for the image. Based on the image, explain why the answer is correct.' + \
'Try to briefly think step by step and use references objects/scales to reason. Then do the following things. Follow the format strictly:\n' + \
'1. Summarize the rationale into a brief one, and output it in the format of [COT: <brief rationale>]' + \
'2. If there are blue bounding boxes with a number i on the top left, tell me what is the bounded region in the format of [DESC(i): <the name of region>]'


completion = client.chat.completions.create(
    model="qwen2.5-vl-7b-instruct",  # 在Body中指明要访问的模型名，即“model”（见2.1表格1）
    messages=[
        {
        "role": "user", 
        "content": [
            {
                "type": "text",
                # "text":  "What is the approximate height of Region [0] in meters? Answer the question in one line."
                # "text":  template.format("Who wears the glasses?", "man")
                "text":  template.format("What the standing woman holds?", "table computer.")
            },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

        ]
    }
    ],
    stream=False,
    max_tokens=1024,
    temperature=0.2
)
print(completion.model_dump_json())

