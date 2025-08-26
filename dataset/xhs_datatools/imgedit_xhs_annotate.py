import os
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
import json
import os
import argparse
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from datasets import load_dataset
import numpy as np
import base64
import aiohttp
import openai
import ast
import glob
from io import BytesIO
from tqdm import tqdm

from dataset.xhs_datatools.xhs_to_mbeir_format import *  

client = openai.AsyncOpenAI(
    api_key="QSTd3138ffe4ce7dbb4248165b848605fc3",
    base_url="http://redservingapi.devops.xiaohongshu.com/v1"
)

completed_tasks_in_batch = 0
total_tasks_in_batch = 1000
task_completion_lock = None

async def report_task_completion():
    global completed_tasks_in_batch, task_completion_lock
    if task_completion_lock is None:
        task_completion_lock = asyncio.Lock()
    
    async with task_completion_lock:
        completed_tasks_in_batch += 1
        if completed_tasks_in_batch % 10 == 0:  # Report every 10 tasks to reduce output
            print(f"Batch progress: {completed_tasks_in_batch}/{total_tasks_in_batch} tasks completed.")

COMMON_KWARGS = {"temperature": 0.8, "max_tokens": 1024, "stream":False}
MODEL = "qwen2.5-vl-72b-instruct"
# MODEL = "qwen2.5-vl-7b-instruct"
# MODEL="glm-4.5"

prompt_text = f'''1.你是一个对比学习数据标注人员，擅长描述不同图片的差异和关系，从而获得query-candidate对。下方输入query物体的图片和一组candidatet图片，基于每张candidatet图，描述query物体在这张图中所处的环境/与周围物体的关系/它本身属性的变化，例如：
a. 一个女人穿着这件上衣，坐在演播室里进行播音
b. 相同的沙发摆在客厅里，前方有一个茶几
c. 绿色瓶子的化妆水，左边坐着一只小猫
d. 同样形状的小熊玩偶，只不过是棕色而非米色，放在一群玩偶中
2. 描述规则：
a. 保持简洁，在100词以内。
b. 有足够的信息量，能够让人理解query物体在candidatet图中的具体情况。
c. 描述要有唯一性，根据query和描述，能匹配到唯一的candidatet图
3. 输出格式：
a. 每张candidate图有一个描述和一个分数（0-1之间，表示描述的相关性和准确性）——如果candidate图片和query过于相似，或者不相似但没有区分性高的参考信息，很难描述，可以给出0分。如果容易的话给出1分
b. 使用英文描述
c. 严格遵守json格式，输出格式如下：
"candidates": [
{{
    "0": "description0",
    "score": "0.7"
}},
{{
    "1": "description1",
    "score": "0.1"
}},
...
]
以下是输入图片
query图片:
'''

class XHSGoodTask7FileExtractor(XHSGoodsFileExtractor):
    """Extractor for XHS goods dataset format"""
    
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "xhs_data/goods_data/from_20250401_to_20250407/images",
                 samples_for_test_up: int = 2000,
                 samples_for_test_low: int = 0,
                 top_k_poscand: int = 10,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, image_subdir, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples)
        self.IMAGE_SUBDIR = image_subdir
    
    def crop_query_image(self, image_path: str, box: Dict) -> Image.Image:
        """Crop query image based on bounding box"""
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # Convert normalized box coordinates to absolute pixel coordinates
            x0 = int(width * box[0])
            y0 = int(height * box[1])
            x1 = int(width * box[2])
            y1 = int(height * box[3])
            
            # Crop the image
            cropped_image = image.crop((x0, y0, x1, y1))
            return cropped_image
        except Exception as e:
            print(f"Error cropping image {image_path}: {e}")
            return None
    
    def draw_boxes_on_image(self, image_path: str, boxes: List[Dict]) -> Image.Image:
        """Draw bounding boxes on image with numbers"""
        try:
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            width, height = image.size
            
            for i, box in enumerate(boxes):
                # Convert normalized box coordinates to absolute pixel coordinates
                x0 = int(width * box[0])
                y0 = int(height * box[1])
                x1 = int(width * box[2])
                y1 = int(height * box[3])
                
                # Draw blue rectangle
                line_thickness = max(2, int(min(width, height) * 0.01))
                draw.rectangle([(x0, y0), (x1, y1)], outline="blue", width=line_thickness)
                
                # Draw number on top left of box
                draw.text((x0, y0-20), str(i), fill="blue", font_size=20)
            
            return image
        except Exception as e:
            print(f"Error drawing boxes on image {image_path}: {e}")
            return None
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        return encoded
    def cal_box_score(self, box) -> float:
        return max(box[2]-box[0],box[3]-box[1])
    
    async def process_with_api(self, query_image: Image.Image, candidate_images: List[Image.Image]) -> Optional[Dict]:
        """Process images with OpenAI API and return parsed JSON result"""
        try:
            print(f"Starting API call with {len(candidate_images)} candidate images...")
            # Encode images to base64
            query_encoded = self.encode_image_to_base64(query_image)
            candidate_encoded_list = [self.encode_image_to_base64(img) for img in candidate_images]
            
            # Build message content with text and images
            content = [{"type": "text", "text": prompt_text}]
            
            # Add query image
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{query_encoded}"}
            })
            
            # Add candidate images
            for i, encoded in enumerate(candidate_encoded_list):
                content.append({"type": "text", "text": "candidate图片:\n"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                })
            
            messages = [{"role": "user", "content": content}]
            
            print(f"Sending request to API...")
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                **COMMON_KWARGS
            )
            result = resp.choices[0].message.content
            print(f"Received API response (length: {len(result)})")
            # breakpoint()
            # result = resp.model_dump_json()["choices"][0]["message"]["content"]
            await report_task_completion()
            # print(resp)
            
            # Try to parse JSON result
            try:
                # Extract JSON from the response
                if '"candidates":' in result:
                    start_idx = result.find('"candidates":')
                    # Find the start of the array
                    json_start = result.find('[', start_idx)
                    if json_start == -1:
                        print(f"Warning: Could not find JSON array start in API response")
                        return None
                    
                    # Find the end of the array by counting brackets
                    bracket_count = 0
                    json_end = json_start
                    for i, char in enumerate(result[json_start:], json_start):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                    
                    json_str = '{"candidates": ' + result[json_start:json_end] + '}'
                    parsed_result = json.loads(json_str)
                    return parsed_result
                else:
                    print(f"Warning: No 'candidates' field found in API response: {result}")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from API response: {e}")
                print(f"Raw response: {result}")
                return None
                
        except Exception as e:
            print(f"Error calling API: {e}")
            return None
    
    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        """Extract queries and candidates from XHS goods format files"""
        # Use asyncio to run the async version
        return asyncio.run(self._extract_queries_and_candidates_async())
    
    async def _process_single_query(self, semaphore: asyncio.Semaphore, original_query_id: str, 
                                   candidate_items_for_query: List, counter: int) -> List[QueryCandidate]:
        """Process a single query with API call"""
        async with semaphore:
            try:
                original_query_image_id = candidate_items_for_query[0].get("query_image_path").split("/")[-1]
                original_query_image_url = candidate_items_for_query[0].get("query_image_url")
                original_query_box = candidate_items_for_query[0].get("query_box")

                # Build score mapping exactly like original
                sorted_original_score_keys_for_current_query_candidates = {}
                for j, dd in enumerate(candidate_items_for_query):
                    sorted_original_score_keys_for_current_query_candidates[j+1] = {
                        'img2img': dd['img2img'],
                        'box_score': self.cal_box_score(dd['doc_box'])
                    }

                # Convert candidate_items_for_query to list of DocCandidate
                doc_candidates = []
                for item in candidate_items_for_query:
                    doc_candidates.append(DocCandidate(
                        doc_image_url=item.get("doc_image_url", None),
                        doc_image_path=item.get("doc_image_path").split("/")[-1],
                        doc_box=item.get("doc_box", None),
                        caption=item.get("caption", None)
                    ))

                candidate_items_for_query = doc_candidates

                if len(candidate_items_for_query) != len(sorted_original_score_keys_for_current_query_candidates):
                    print(f"Warning: Mismatch in candidate list length ({len(candidate_items_for_query)}) and score key count ({len(sorted_original_score_keys_for_current_query_candidates)}) for query '{original_query_id}'. Skipping this query.")
                    return []

                temp_pos_candidates_scores = []
                temp_neg_original_score_keys = []

                # Iterate based on the sorted score keys exactly like original
                for original_score_key, scores_dict in sorted_original_score_keys_for_current_query_candidates.items():
                    img_score = scores_dict.get('img2img', 0)
                    box_score = scores_dict.get('box_score', 0)
                    
                    if img_score > 1:  # Positive candidate
                        temp_pos_candidates_scores.append((original_score_key, img_score, box_score))
                    else:
                        temp_neg_original_score_keys.append(original_score_key)
                
                # NEW FILTER: Check if we have at least 2 candidates with box_score > 0.8 or img2img > 1
                high_quality_count = 0
                for original_score_key, scores_dict in sorted_original_score_keys_for_current_query_candidates.items():
                    img_score = scores_dict.get('img2img', 0)
                    box_score = scores_dict.get('box_score', 0)
                    if box_score <0.85 and img_score > 1:
                        high_quality_count += 1
                
                if high_quality_count < 2:
                    print(f"Skipping query {original_query_id}: only {high_quality_count} high-quality candidates (need at least 2)")
                    return []
                
                if not temp_pos_candidates_scores:
                    return []
                elif len(temp_pos_candidates_scores) == len(candidate_items_for_query):
                    return []

                # Sort positive candidates by score exactly like original
                temp_pos_candidates_scores.sort(key=lambda x: (-x[1], -x[2]))
                
                # Get top 3 positive candidates for API processing
                if len(temp_pos_candidates_scores) < 3:
                    top_3_pos_candidates = temp_pos_candidates_scores
                else:
                    top_3_pos_candidates = temp_pos_candidates_scores[:3]
                
                # Get query image path
                query_image_path = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR, original_query_image_id)
                if not os.path.exists(query_image_path):
                    print(f"Query image not found: {query_image_path}")
                    return []

                cropped_query = self.crop_query_image(query_image_path, original_query_box)
                
                # Get top 3 candidate images and their boxes
                candidate_images = []
                candidate_boxes = []
                for score_key, _, _ in top_3_pos_candidates:
                    candidate_idx = score_key - 1  # Convert to 0-based index
                    candidate_item = candidate_items_for_query[candidate_idx]
                    
                    candidate_image_path = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR, candidate_item.doc_image_path)
                    if os.path.exists(candidate_image_path):
                        # Draw boxes on candidate image
                        boxes_to_draw = [candidate_item.doc_box] if candidate_item.doc_box else []
                        candidate_with_box = self.draw_boxes_on_image(candidate_image_path, boxes_to_draw)
                        if candidate_with_box:
                            candidate_images.append(candidate_with_box)
                            candidate_boxes.append(candidate_item.doc_box)
                
                # Call API asynchronously
                print(f"Calling API for query {original_query_id} with {len(candidate_images)} candidate images...")
                api_result = await self.process_with_api(cropped_query, candidate_images)
                print(f"API result for {original_query_id}: {api_result}")
                if api_result is None:
                    print(f"API call failed for query {original_query_id}")
                    return []
                
                # Process API result and create new QueryCandidate objects
                candidates_data = api_result.get('candidates', [])
                high_score_descriptions = []
                
                for cand_data in candidates_data:
                    if isinstance(cand_data, dict):
                        score = float(cand_data.get('score', 0))
                        if score > 0.5:
                            description = cand_data.get(str(len(high_score_descriptions)), '')
                            if description:
                                high_score_descriptions.append(description)
                
                # Create QueryCandidate objects if we have at least 2 high-score descriptions
                result_query_candidates = []
                if len(high_score_descriptions) >= 2:
                    for i, description in enumerate(high_score_descriptions):
                        if i < len(top_3_pos_candidates):
                            score_key, _, _ = top_3_pos_candidates[i]
                            selected_candidate = candidate_items_for_query[score_key-1]
                            
                            # Build complete candidate_items list: positive + all negatives
                            complete_candidate_items = [selected_candidate]  # Start with the positive candidate
                            
                            # Add all negative candidates
                            for neg_key in temp_neg_original_score_keys:
                                neg_candidate = candidate_items_for_query[neg_key-1]
                                complete_candidate_items.append(neg_candidate)
                            
                            # Adjust neg_keys to match the new candidate_items indices
                            adjusted_neg_keys = list(range(2, len(complete_candidate_items) + 1))  # Start from 2 since 1 is positive
                            
                            new_query_candidate = QueryCandidate(
                                query_id=f"{original_query_id}_api_{i}",
                                query_image_id=original_query_image_id,
                                query_image_url=original_query_image_url,
                                query_box=original_query_box,
                                query_txt=description,  # Use API description as query text
                                candidate_items=complete_candidate_items,
                                best_pos_keys=[1],  # First item is positive
                                neg_keys=adjusted_neg_keys  # All other items are negative
                            )
                            print("debug@@@@:\n", new_query_candidate.query_image_id, selected_candidate.doc_image_path)
                            result_query_candidates.append(new_query_candidate)
                else:
                    print(f"Insufficient high-score descriptions ({len(high_score_descriptions)}) for query {original_query_id}")
                    
                return result_query_candidates
                    
            except Exception as e:
                print(f"Error processing query {original_query_id}: {e}")
                return []

    async def _extract_queries_and_candidates_async(self) -> List[QueryCandidate]:
        """Extract queries and candidates from XHS goods format files with concurrent processing"""
        
        samples_path = os.path.join(self.ROOT_FS, self.samples_path)
        
        with open(samples_path, 'r', encoding='utf-8') as f:
            queries_input = json.load(f)
        CONCUR=100
        
        # Create semaphore to limit concurrency to 10
        semaphore = asyncio.Semaphore(CONCUR)
        
        # Prepare tasks for concurrent execution
        tasks = []
        valid_query_count = 0
        
        for counter, (original_query_id, candidate_items_for_query) in enumerate(queries_input.items()):
            # Skip samples based on SKIP_SAMPLES
            if counter < self.SKIP_SAMPLES:
                continue
                
            # Apply sampling limits using instance variables
            if valid_query_count >= self.SAMPLES_FOR_TEST_UP:
                break
            if valid_query_count < self.SAMPLES_FOR_TEST_LOW:
                valid_query_count += 1
                continue
                
            # Create task for this query
            task = self._process_single_query(semaphore, original_query_id, candidate_items_for_query, counter)
            tasks.append(task)
            valid_query_count += 1
        
        print(f"Starting concurrent processing of {len(tasks)} queries with concurrency limit of 10...")
        print(f"Processing queries from index {self.SAMPLES_FOR_TEST_LOW} to {min(valid_query_count, self.SAMPLES_FOR_TEST_UP)}")
        
        # Execute all tasks concurrently and collect results
        all_query_candidates = []
        completed_tasks = 0
        # Process tasks in batches to avoid overwhelming the system
        for i in range(0, len(tasks), CONCUR):  # Process in batches of CONCUR
            batch_tasks = tasks[i:i+CONCUR]
            print(f"Processing batch {i//CONCUR + 1}/{(len(tasks)-1)//CONCUR + 1} with {len(batch_tasks)} tasks...")
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Task failed with exception: {result}")
                else:
                    all_query_candidates.extend(result)
                completed_tasks += 1
            
            # the checkpoint system
            if i % 1000 == 0:
                import pickle
                with open(f"log.pkl", "wb") as f:
                    pickle.dump(all_query_candidates, f)

                
            print(f"Completed {completed_tasks}/{len(tasks)} queries, collected {len(all_query_candidates)} query candidates so far")
        
        print(f"Finished processing all queries. Total query candidates: {len(all_query_candidates)}")
        return all_query_candidates

def parquet_filter(
    outfile,
    parquet_path
    # parquet_path = "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/goods_data/from_20250316_to_20250607/all_data_with_valid_human_label_v2.0.parquet"
):
    '''把alldata_with human valid的parquet过滤album，camera等，然后转json
    '''
    import pandas as pd
    import json

    df = pd.read_parquet(parquet_path, engine="pyarrow")

    exclude_sources = ["album", "camera", "goods", "goods_detail"]
    df_filtered = df[~df["search_source"].isin(exclude_sources)]

    df_filtered["clean_key"] = df_filtered["unique_key"].str.split("@").str[0]
    grouped = df_filtered.groupby("clean_key").size().reset_index(name="count")
    grouped = grouped.sort_values(by="count", ascending=False)
    df_with_count = df_filtered.merge(grouped, on="clean_key", how="left")
    df_sorted = df_with_count.sort_values(by=["count", "clean_key"], ascending=[False, True])

    # 如果不需要count列，可以删除
    # df_sorted = df_sorted.drop("count", axis=1)

    # # 保存为 parquet
    # output_path = "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/goods_data/from_20250316_to_20250607/original_data_sorted_by_group_count.parquet"
    # df_sorted.to_parquet(output_path, index=False, engine="pyarrow")

    # print(f"已生成按分组结果排序后的原始数据 Parquet 文件：{output_path}")
    print(f"原始数据行数: {len(df_sorted)}")
    print(f"clean_key 数量: {df_sorted['clean_key'].nunique()}")
    print("\n前几行数据:")
    print(df_sorted.head())

    grouped = {}

    def cal_box_score(self, box) -> float:
        return max(box[2]-box[0],box[3]-box[1])

    for i,row in df_sorted.iterrows():
        clear_key = row.get('clean_key') or row.get('key')
        if clear_key not in grouped:
            grouped[clear_key] = []

        boxq = row["query_box"].tolist()
        boxd = row["doc_box"].tolist()
        if cal_box_score(1, boxq) >0.9 or cal_box_score(1, boxd) >0.9:
            continue
        # 构造每条doc的数据
        entry = {
            "search_source": row["search_source"],
            "query_image_url": row["query_image_url"],
            "query_box": row["query_box"].tolist(),
            "query_text": row["query_text"],
            "doc_image_url": row["doc_image_url"],
            "doc_box": row["doc_box"].tolist(),
            "doc_text": row["doc_text"],
            "img2txt": float(row["img2txt@human"]),
            "img2img": float(row["img2img@human"]),
            "mm2mm": float(row["mm2mm@human"]),
            "query_image_path": row["query_image_path"],
            "query_image_oss": row["query_image_oss"],
            "doc_image_path": row["doc_image_path"],
            "doc_image_oss": row["doc_image_oss"]
        }
        grouped[clear_key].append(entry)
    # 输出为list，每组加上key
    # result_list = []
    # for k,v in grouped.items():
    #     result_list.append({k: v})

    # 或直接用dict，见 grouped
    with open(outfile, "w") as f:
        json.dump(grouped, f, indent=4, ensure_ascii=False)


class NewXHSGoodTask7FileExtractor(XHSGoodTask7FileExtractor):
    def __init__(self, samples_path, labels_path = "", root_fs = "/mnt/tidal-alsh01/dataset/mmeb/", image_subdir = "xhs_data/goods_data/from_20250401_to_20250407/images", samples_for_test_up = 2000, samples_for_test_low = 0, top_k_poscand = 10, skip_samples = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, image_subdir, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples, **kwargs)
    
    async def download_image_from_url(self, url: str, save_path: str) -> bool:
        """Download image from URL and save to local path"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(save_path, 'wb') as f:
                            f.write(await response.read())
                        print(f"Downloaded image from {url} to {save_path}")
                        return True
                    else:
                        print(f"Failed to download image from {url}: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return False
    
    async def ensure_image_exists(self, image_path: str, image_url: str) -> bool:
        """Ensure image exists locally, download if not"""
        if os.path.exists(image_path):
            return True
        
        if image_url:
            return await self.download_image_from_url(image_url, image_path)
        
        print(f"Image not found and no URL provided: {image_path}")
        return False
    
    async def _process_single_query(self, semaphore: asyncio.Semaphore, original_query_id: str, 
                                   candidate_items_for_query: List, counter: int) -> List[QueryCandidate]:
        """Process a single query with API call"""
        async with semaphore:
            try:
                original_query_image_id = candidate_items_for_query[0].get("query_image_path")
                original_query_image_url = candidate_items_for_query[0].get("query_image_url")
                original_query_box = candidate_items_for_query[0].get("query_box")

                # Build score mapping exactly like original
                sorted_original_score_keys_for_current_query_candidates = {}
                for j, dd in enumerate(candidate_items_for_query):
                    sorted_original_score_keys_for_current_query_candidates[j+1] = {
                        'img2img': dd['img2img'],
                        'box_score': self.cal_box_score(dd['doc_box'])
                    }

                # Convert candidate_items_for_query to list of DocCandidate
                doc_candidates = []
                for item in candidate_items_for_query:
                    doc_candidates.append(DocCandidate(
                        doc_image_url=item.get("doc_image_url", None),
                        doc_image_path=item.get("doc_image_path"),
                        doc_box=item.get("doc_box", None),
                        caption=item.get("caption", None)
                    ))

                candidate_items_for_query = doc_candidates

                if len(candidate_items_for_query) != len(sorted_original_score_keys_for_current_query_candidates):
                    print(f"Warning: Mismatch in candidate list length ({len(candidate_items_for_query)}) and score key count ({len(sorted_original_score_keys_for_current_query_candidates)}) for query '{original_query_id}'. Skipping this query.")
                    return []

                temp_pos_candidates_scores = []
                temp_neg_original_score_keys = []

                # Iterate based on the sorted score keys exactly like original
                for original_score_key, scores_dict in sorted_original_score_keys_for_current_query_candidates.items():
                    img_score = scores_dict.get('img2img', 0)
                    box_score = scores_dict.get('box_score', 0)
                    
                    if img_score > 1:  # Positive candidate
                        temp_pos_candidates_scores.append((original_score_key, img_score, box_score))
                    else:
                        temp_neg_original_score_keys.append(original_score_key)
                
                # NEW FILTER: Check if we have at least 2 candidates with box_score > 0.8 or img2img > 1
                high_quality_count = 0
                for original_score_key, scores_dict in sorted_original_score_keys_for_current_query_candidates.items():
                    img_score = scores_dict.get('img2img', 0)
                    box_score = scores_dict.get('box_score', 0)
                    if box_score <0.85 and img_score > 1:
                        high_quality_count += 1
                
                if high_quality_count < 2:
                    print(f"Skipping query {original_query_id}: only {high_quality_count} high-quality candidates (need at least 2)")
                    return []
                
                if not temp_pos_candidates_scores:
                    return []
                elif len(temp_pos_candidates_scores) == len(candidate_items_for_query):
                    return []

                # Sort positive candidates by score exactly like original
                temp_pos_candidates_scores.sort(key=lambda x: (-x[1], -x[2]))
                
                # Get top 3 positive candidates for API processing
                if len(temp_pos_candidates_scores) < 3:
                    top_3_pos_candidates = temp_pos_candidates_scores
                else:
                    top_3_pos_candidates = temp_pos_candidates_scores[:3]
                
                # Get query image path and ensure it exists
                query_image_path = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR, original_query_image_id)
                query_exists = await self.ensure_image_exists(query_image_path, original_query_image_url)
                if not query_exists:
                    print(f"Query image not found and could not be downloaded: {query_image_path}")
                    return []

                cropped_query = self.crop_query_image(query_image_path, original_query_box)
                
                # Get top 3 candidate images and their boxes
                candidate_images = []
                candidate_boxes = []
                for score_key, _, _ in top_3_pos_candidates:
                    candidate_idx = score_key - 1  # Convert to 0-based index
                    candidate_item = candidate_items_for_query[candidate_idx]
                    
                    candidate_image_path = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR, candidate_item.doc_image_path)
                    
                    # Ensure candidate image exists, download if needed
                    candidate_exists = await self.ensure_image_exists(candidate_image_path, candidate_item.doc_image_url)
                    if candidate_exists:
                        # Draw boxes on candidate image
                        boxes_to_draw = [candidate_item.doc_box] if candidate_item.doc_box else []
                        candidate_with_box = self.draw_boxes_on_image(candidate_image_path, boxes_to_draw)
                        if candidate_with_box:
                            candidate_images.append(candidate_with_box)
                            candidate_boxes.append(candidate_item.doc_box)
                
                # Call API asynchronously
                print(f"Calling API for query {original_query_id} with {len(candidate_images)} candidate images...")
                api_result = await self.process_with_api(cropped_query, candidate_images)
                print(f"API result for {original_query_id}: {api_result}")
                if api_result is None:
                    print(f"API call failed for query {original_query_id}")
                    return []
                
                # Process API result and create new QueryCandidate objects
                candidates_data = api_result.get('candidates', [])
                high_score_descriptions = []
                
                for cand_data in candidates_data:
                    if isinstance(cand_data, dict):
                        score = float(cand_data.get('score', 0))
                        if score > 0.5:
                            description = cand_data.get(str(len(high_score_descriptions)), '')
                            if description:
                                high_score_descriptions.append(description)
                
                # Create QueryCandidate objects if we have at least 2 high-score descriptions
                result_query_candidates = []
                if len(high_score_descriptions) >= 2:
                    for i, description in enumerate(high_score_descriptions):
                        if i < len(top_3_pos_candidates):
                            score_key, _, _ = top_3_pos_candidates[i]
                            selected_candidate = candidate_items_for_query[score_key-1]
                            
                            # Build complete candidate_items list: positive + all negatives
                            complete_candidate_items = [selected_candidate]  # Start with the positive candidate
                            
                            # Add all negative candidates
                            for neg_key in temp_neg_original_score_keys:
                                neg_candidate = candidate_items_for_query[neg_key-1]
                                complete_candidate_items.append(neg_candidate)
                            
                            # Adjust neg_keys to match the new candidate_items indices
                            adjusted_neg_keys = list(range(2, len(complete_candidate_items) + 1))  # Start from 2 since 1 is positive
                            
                            new_query_candidate = QueryCandidate(
                                query_id=f"{original_query_id}_api_{i}",
                                query_image_id=original_query_image_id,
                                query_image_url=original_query_image_url,
                                query_box=original_query_box,
                                query_txt=description,  # Use API description as query text
                                candidate_items=complete_candidate_items,
                                best_pos_keys=[1],  # First item is positive
                                neg_keys=adjusted_neg_keys  # All other items are negative
                            )
                            print("debug@@@@:\n", new_query_candidate.query_image_id, selected_candidate.doc_image_path)
                            result_query_candidates.append(new_query_candidate)
                else:
                    print(f"Insufficient high-score descriptions ({len(high_score_descriptions)}) for query {original_query_id}")
                    
                return result_query_candidates
                    
            except Exception as e:
                print(f"Error processing query {original_query_id}: {e}")
                return []


# generate_mbeir_files(
#     "xhs_data/goods_data/from_20250401_to_20250407/from_20250401_to_20250407.json",
#     "",
#     "M-BEIR",
#     dsname="xhsgood",
#     cls = XHSGoodTask7FileExtractor,
#     dataset_id="10", task_id="7",
#     root_fs=ROOT_FS,
#     image_subdir="xhs_data/goods_data/from_20250401_to_20250407/images",
#     samples_for_test_up=1000, samples_for_test_low=0, top_k_poscand=10,
#     skip_samples=0
# )


# TRAIN = True

# generate_mbeir_files(
#     "xhs_data/goods_data/from_20250401_to_20250407/from_20250401_to_20250407.json",
#     "",
#     "M-BEIR",
#     dsname="xhsgoodmoreneg2",
#     cls = XHSGoodTask7FileExtractor,
#     dataset_id="10", task_id="7",
#     root_fs=ROOT_FS,
#     image_subdir="xhs_data/goods_data/from_20250401_to_20250407/images",
#     samples_for_test_up=3000, samples_for_test_low=0, top_k_poscand=10,
#     skip_samples=3000
# )


# NOTE! 要把TRAIN 设为True
generate_mbeir_files(
    "xhs_data/goods_data/from_20250316_to_20250607/v2.json",
    "",
    "M-BEIR",
    dsname="xhsgoodtrain",
    cls = NewXHSGoodTask7FileExtractor,
    dataset_id="10", task_id="7",
    root_fs=ROOT_FS,
    image_subdir="xhs_data/goods_data/from_20250316_to_20250607/images",
    samples_for_test_up=3500, samples_for_test_low=0, top_k_poscand=10,
    skip_samples=0
)

generate_mbeir_files(
    "xhs_data/goods_data/from_20250316_to_20250607/v4.json",
    "",
    "M-BEIR",
    dsname="xhsgoodtrainv4",
    cls = NewXHSGoodTask7FileExtractor,
    dataset_id="10", task_id="7",
    root_fs=ROOT_FS,
    image_subdir="xhs_data/goods_data/from_20250316_to_20250607/images",
    samples_for_test_up=9000, samples_for_test_low=0, top_k_poscand=10,
    skip_samples=0
)

# /mnt/tidal-alsh01/usr/liangxun/ICLR26/dam-qwen2-cp/dataset/xhs_datatools/imgedit_xhs_annotate.py