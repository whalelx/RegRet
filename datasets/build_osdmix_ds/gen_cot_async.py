import os
import base64
import requests
from io import BytesIO
import argparse, json, datetime, time, asyncio
import aiohttp
import openai
from tqdm import tqdm
client = OpenAI(
    api_key="",
    base_url=""
)

completed_tasks_in_batch = 0
total_tasks_in_batch = 99000

async def report_task_completion():
    global completed_tasks_in_batch
    completed_tasks_in_batch += 1
    print(f"Batch progress: {completed_tasks_in_batch}/{total_tasks_in_batch} tasks completed.")

COMMON_KWARGS = {"temperature": 0.2, "max_tokens": 1024, "stream":False}
MODEL = "qwen2.5-vl-72b-instruct"
# MODEL = "qwen2.5-vl-32b-instruct"
def genquery(question, answer):
    return f'There is a question "{question}" and a correct answer "{answer}" for the image. Based on the image, explain why the answer is correct.' + \
'Try to briefly think step by step and use references objects/scales to reason. Then do the following things. Follow the format strictly:\n' + \
'1. Summarize the rationale into a brief one, and output it in the format of [COT: <brief rationale>]' + \
'2. If there are blue bounding boxes with a number i on the top left, use a few nouns to tell me what is the the box\'s content? Use the format of [DESC(i): <the name of region>]'

async def call_chatgpt_async(key, question, answer, image_url):
    query = genquery(question, answer)
    with open(image_url, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
        ]},
    ]

    try:
        resp = await client.chat.completions.create(
            model=MODEL,  # Use the correct model name
            messages=messages,  # Provide the prompt directly
            **COMMON_KWARGS
        )
        # print(resp)
        result = resp.choices[0].message.content
        await report_task_completion() #NOTE
    except Exception as e:
        print(f"Error: {e}")
        result = ""
    return key, result

async def call_chatgpt_bulk(keys, questions, answers, imgs):
    # gather a bunch of acreate() calls concurrently
    tasks = [
        call_chatgpt_async(k, q, a, img)
        for k, q, a, img in zip(keys, questions, answers, imgs)
    ]
    return await asyncio.gather(*tasks)

def bulk_evaluate(data, batch_size):
    output = []
    keys = [i for i in range(batch_size)]
    for start in tqdm(range(109, len(data), batch_size)):
        batch = data[start : start + batch_size]

        imgs = [d["images"][0]                                                       for d in batch]
        ans = [d["messages"][-1]["content"].split('\n')[-1]                         for d in batch]
        qs = [d["messages"][0]["content"].split('\n')[1].split("Identify")[0]       for d in batch]

        responses = asyncio.run(call_chatgpt_bulk(keys, qs, ans, imgs))
        currlst = []
        for key, res in responses:
            currlst.append(tuple([key, {"cot":res}]))
            currlst.sort(key=lambda x: x[0])
        for d,res in zip(batch, currlst):
            d.update(res[1])

        outfile = "./osd-vpt-cot.json"
        if not os.path.exists(outfile):
            all_sampled_data=[]
        else:
            with open(outfile, "r", encoding='utf-8') as f:
                all_sampled_data = json.load(f)
        with open(outfile, "w", encoding='utf-8') as f:
            all_sampled_data.extend(batch)
            print(len(all_sampled_data))
            json.dump(all_sampled_data, f, indent=4, ensure_ascii=False)
        # output.extend(currlst)
        # time.sleep(1)  # rate‐limit if you need

    return output

if __name__ == "__main__":
    with open("/mnt/tidal-alsh01/usr/liangxun/STAUG/LLaMA-Factory/data/osd-vpt.json","r") as f:
        data = json.load(f)
    bulk_evaluate(data, 1000)