import torch
from PIL import Image
import requests
from io import BytesIO
import json
from tqdm import tqdm
import os
import asyncio
from openai import AsyncOpenAI
import base64  # 1. 导入 base64 库

API_BASE = "https://api2.aigcbest.top/v1/"#chat/completions"
API_KEY = ""
MODEL = "gpt-4o-2024-11-20"

# 2. 新增辅助函数：将图片文件编码为 Base64 Data URI
def encode_image_to_base64_data_uri(image_path: str) -> str:
    """
    读取本地图片，编码为 Base64 并格式化为 Data URI.
    """
    try:
        # 使用 PIL 打开图片以确定其格式 (JPEG, PNG, etc.)
        with Image.open(image_path) as image:
            mime_type = image.format
            if mime_type is None: # 如果无法确定格式，默认使用 jpeg
                mime_type = 'jpeg'
            
            # 使用 BytesIO 将图片保存在内存中
            buffered = BytesIO()
            # 保存时明确指定格式，以防原始文件格式有问题
            image.save(buffered, format=mime_type)
            img_bytes = buffered.getvalue()

        # 将字节编码为 Base64 字符串
        base64_encoded_data = base64.b64encode(img_bytes)
        base64_image_string = base64_encoded_data.decode('utf-8')

        # 格式化为 Data URI
        return f"data:image/{mime_type.lower()};base64,{base64_image_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        raise

async def batch_predict_from_sharegpt(
    input_sharegpt_path: str,
    output_jsonl_path: str,
    api_base_url: str=API_BASE,
    api_key: str=API_KEY,
    model_name: str = MODEL,
    batch_size: int = 8,
    max_new_tokens: int = 512
):
    batch_size=2
    
    print(f"Initializing API client for base_url: {api_base_url}")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url
    )

    all_input_data = []
    print(f"Reading input data from: {input_sharegpt_path}...")
    try:
        with open(input_sharegpt_path, 'r', encoding='utf-8') as f_in:
            all_input_data.extend(json.load(f_in))
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_sharegpt_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_sharegpt_path}. Ensure it's a valid JSON array.")
        return

    if not all_input_data:
        print("No valid data found in input file.")
        return

    print(f"Starting API batch processing for {len(all_input_data)} items with concurrency {batch_size}...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(all_input_data), batch_size), desc="Processing Batches"):
            batch_items = all_input_data[i:i+batch_size]
            
            tasks = []
            original_ids_in_batch = []
            
            for item_idx, item in enumerate(batch_items):
                image_path = item.get("images")[0] if "images" in item and item["images"] else None
                prompt_content = item.get("messages")[0]["content"] if "messages" in item and item["messages"] else "Describe the image."
                item_id = item.get("index", f"item_{i+item_idx}")
                
                # 3. 准备图片内容，区分 URL 和本地路径
                image_content_for_api = None
                if image_path:
                    try:
                        if image_path.startswith("http"):
                            # 如果是 URL，直接使用
                            image_content_for_api = {
                                "type": "image_url",
                                "image_url": {"url": image_path}
                            }
                        else:
                            # 如果是本地路径，编码为 Base64 Data URI
                            if not os.path.exists(image_path):
                                print(f"Warning: Local image path not found: {image_path} for item id '{item_id}'. Skipping.")
                                continue
                            data_uri = encode_image_to_base64_data_uri(image_path)
                            image_content_for_api = {
                                "type": "image_url",
                                "image_url": {"url": data_uri}
                            }
                    except Exception as e:
                        print(f"Warning: Failed to process image {image_path} for item id '{item_id}': {e}. Skipping.")
                        continue

                original_ids_in_batch.append(item_id)
                
                messages_for_api = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_content + "Answer in brief!"}
                        ]
                    }
                ]
                
                # 如果图片处理成功，则添加到消息中
                if image_content_for_api:
                    messages_for_api[0]["content"].append(image_content_for_api)

                task = client.chat.completions.create(
                    model=model_name,
                    messages=messages_for_api,
                    max_tokens=max_new_tokens,
                )
                tasks.append(task)

            if not tasks:
                continue

            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"An unexpected error occurred during asyncio.gather: {e}")
                continue

            for item_id, response in zip(original_ids_in_batch, responses):
                if isinstance(response, Exception):
                    predict_text = f"Error: API call failed - {response}"
                    print(f"ID {item_id}: {predict_text}")
                else:
                    predict_text = response.choices[0].message.content.strip()
                    print(predict_text)
                
                f_out.write(json.dumps({"id": item_id, "predict": predict_text}) + "\n")
            
            f_out.flush()

    print(f"API batch processing complete. Output saved to: {output_jsonl_path}")

# --- 主函数入口 ---
if __name__ == '__main__':
    INPUT_FILE = "your_input_data.jsonl"
    OUTPUT_FILE = "predictions_output.jsonl"
    API_BASE = "https://api2.aigcbest.top/v1"
    API_KEY = "YOUR_API_KEY"  # 替换为您的 API Key
    MODEL = "gpt-4o-2024-11-20"
    CONCURRENCY = 16

    # 4. 更新示例文件创建逻辑，包含一个本地图片
    if not os.path.exists(INPUT_FILE):
        print(f"Creating a sample input file: {INPUT_FILE}")
        
        # 下载一张图片作为本地文件示例
        local_image_path = "cat_logo.png"
        image_url = "https://github.com/dianping/cat/raw/master/cat-home/src/main/webapp/images/logo/cat_logo03.png"
        if not os.path.exists(local_image_path):
            print(f"Downloading sample image to {local_image_path}...")
            try:
                res = requests.get(image_url)
                res.raise_for_status()
                with open(local_image_path, "wb") as f:
                    f.write(res.content)
            except Exception as e:
                print(f"Could not download sample image: {e}")
                local_image_path = None # 下载失败则不使用

        sample_data = []
        # 添加使用本地图片的示例
        if local_image_path:
            sample_data.append({
                "index": "local_cat_logo",
                "images": [local_image_path],  # 使用本地路径
                "messages": [{"role": "user", "content": "这张本地图片是什么？请详细描述。"}],
            })
        
        # 添加使用 URL 的示例
        sample_data.append({
            "index": "remote_clion_screenshot",
            "images": ["https://www.javatiku.cn/wp-content/uploads/2023/09/image-13.png"], # 使用 URL
            "messages": [{"role": "user", "content": "这张网络图片里是什么软件的截图？它在做什么操作？"}]
        })

        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)

    # 运行异步主函数
    asyncio.run(batch_predict_from_sharegpt_api(
        input_sharegpt_path=INPUT_FILE,
        output_jsonl_path=OUTPUT_FILE,
        api_base_url=API_BASE,
        api_key=API_KEY,
        model_name=MODEL,
        batch_size=CONCURRENCY
    ))