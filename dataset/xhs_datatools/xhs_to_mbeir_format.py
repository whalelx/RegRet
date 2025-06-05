import json
import os
import argparse
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm
import asyncio

# Constants for M-BEIR format
DATASET_ID_XHS = "10"
TASK_ID_XHS = "7"
# "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/mbeir_images/xhs_images"
XHS_IMAGE_SUBDIR = "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/note_data/20250304/images"

ON_SERVER = True
DOWNLOAD_IMAGE = False
TRAIN = True

CONCURRENCY= 100

def download_and_draw_box(url, filename, box=None):
    if not ON_SERVER or not DOWNLOAD_IMAGE:
        return
    if os.path.exists(filename):
        return

    response = requests.get(url, stream=True)
    # if the img can not be download, use a white img as a negative candi instead 
    if response.status_code == 403:
        size = (64, 64)
        white_img = Image.new("RGB", size, (255, 255, 255))
        white_img.save(filename)
        print(response)
        return False

    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    img = Image.open(response.raw)
    
    # Convert image to RGB if it's not already (e.g., to handle RGBA, P modes)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.save(filename)
    return True

def generate_mbeir_files(filtered_samples_path, filtered_labels_path, output_dir):
    """
    Generates M-BEIR evaluation files from the outputs of filter_our_test1k.py.
    """
    with open(filtered_samples_path, 'r', encoding='utf-8') as f:
        queries_input = json.load(f)
    with open(filtered_labels_path, 'r', encoding='utf-8') as f:
        labels_input = json.load(f)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    qrels_lines = []
    queries_jsonl_lines = []
    
    global_orig_cand_to_mbeir_did = {}
    global_mbeir_cand_contents = {}
    
    query_counter = 1
    cand_counter = 1

    for original_query_id, candidate_items_for_query in tqdm(queries_input.items()):
        if original_query_id not in labels_input:
            print(f"Warning: Query ID {original_query_id} found in samples but not in labels. Skipping.")
            continue

        original_query_image_id = candidate_items_for_query[0].get("query_image_id").split("/")[-1]
        original_query_image_url = candidate_items_for_query[0].get("query_image_url")
        original_query_box = candidate_items_for_query[0].get("query_box")

        scores_for_query = labels_input[original_query_id]
        # Ensure the order of candidates matches the order of their original score keys
        # This is crucial for correctly associating items from queries_input with their scores
        sorted_original_score_keys_for_current_query_candidates = sorted(scores_for_query.keys(), key=int)

        if len(candidate_items_for_query) != len(sorted_original_score_keys_for_current_query_candidates):
            print(f"Warning: Mismatch in candidate list length ({len(candidate_items_for_query)}) and score key count ({len(sorted_original_score_keys_for_current_query_candidates)}) for query '{original_query_id}'. Skipping this query.")
            continue

        temp_pos_candidates_scores = [] # (original_score_key, img_score, mm_score)
        temp_neg_original_score_keys = []

        # Iterate based on the sorted score keys to ensure correct association with candidate_items_for_query
        for original_score_key in sorted_original_score_keys_for_current_query_candidates:
            scores_dict = scores_for_query[original_score_key]
            img_score = scores_dict.get('img2img', 0)
            mm_score = scores_dict.get('mm2mm', 0)
            
            if img_score > 1: # Positive candidate based on filter_our_test1k.py logic
                temp_pos_candidates_scores.append((original_score_key, img_score, mm_score))
            elif img_score < 1 and mm_score < 1: # Negative candidate
                temp_neg_original_score_keys.append(original_score_key)
        
        if not temp_pos_candidates_scores:
            # print(f"Info: No positive candidates found for query {original_query_id} after score filtering. Skipping query.")
            continue
        elif len(temp_pos_candidates_scores) == len(candidate_items_for_query):
            # if all items are positive, it is too easy
            continue

        temp_pos_candidates_scores.sort(key=lambda x: (-x[1], -x[2])) # Sort by img_score desc, then mm_score desc
        best_pos_original_score_key = temp_pos_candidates_scores[0][0]

        mbeir_qid = f"{DATASET_ID_XHS}:{query_counter}"
        mbeir_pos_cand_did_str = None
        mbeir_neg_cand_dids_list = []

        # Iterate through candidate_items_for_query using the sorted_original_score_keys for indexing
        for i, original_score_key_for_item in enumerate(sorted_original_score_keys_for_current_query_candidates):
            cand_item_dict = candidate_items_for_query[i] # Assuming candidate_items_for_query is ordered like sorted_original_score_keys
            
            original_cand_img_url = cand_item_dict.get("doc_image_url")
            original_cand_img_id = cand_item_dict.get("doc_image_id")
            original_cand_caption = None #cand_item_dict.get("caption")
            original_cand_img_path = original_cand_img_id+".jpg"

            unique_cand_key_part = ""
            if original_cand_img_path:
                unique_cand_key_part = os.path.basename(original_cand_img_path)
            elif original_cand_caption:
                unique_cand_key_part = original_cand_caption[:50] # Truncate for brevity as a key part
            else:
                # Fallback if no image path or caption; less ideal for global uniqueness
                unique_cand_key_part = f"unidentified_cand_oqid_{original_query_id}_osk_{original_score_key_for_item}"
            
            # This identifier is used to map original candidates to unique M-BEIR DIDs.
            # It should be globally unique for each distinct candidate item.
            original_cand_identifier = unique_cand_key_part 

            if original_cand_identifier not in global_orig_cand_to_mbeir_did:
                mbeir_cand_did = f"{DATASET_ID_XHS}:{cand_counter}"
                global_orig_cand_to_mbeir_did[original_cand_identifier] = mbeir_cand_did
                
                cand_img_path_for_pool = None
                if original_cand_img_path:
                    cand_img_path_for_pool = os.path.join(XHS_IMAGE_SUBDIR, os.path.basename(original_cand_img_path))

                cand_txt_for_pool = original_cand_caption
                
                modality = "unknown"
                if cand_img_path_for_pool and cand_txt_for_pool:
                    modality = "image,text"
                elif cand_img_path_for_pool:
                    modality = "image"
                elif cand_txt_for_pool:
                    modality = "text"
                if cand_img_path_for_pool is not None:
                    download_and_draw_box(original_cand_img_url, cand_img_path_for_pool)

                cand_content_for_pool = {
                    "img_path": cand_img_path_for_pool,
                    "modality": modality,
                    "did": mbeir_cand_did,
                    "box": cand_item_dict.get("doc_box", None),
                    "src_content": json.dumps({
                        "original_id": original_cand_identifier, 
                        "original_query_id_context": original_query_id, 
                        "original_score_key_for_query": original_score_key_for_item
                    }),
                    "txt": cand_txt_for_pool
                }
                global_mbeir_cand_contents[mbeir_cand_did] = cand_content_for_pool
                cand_counter += 1
            else:
                mbeir_cand_did = global_orig_cand_to_mbeir_did[original_cand_identifier]

            if original_score_key_for_item == best_pos_original_score_key:
                mbeir_pos_cand_did_str = mbeir_cand_did
            elif original_score_key_for_item in temp_neg_original_score_keys:
                mbeir_neg_cand_dids_list.append(mbeir_cand_did)

        if mbeir_pos_cand_did_str:
            qrels_lines.append(f"{mbeir_qid} 0 {mbeir_pos_cand_did_str} 1 {TASK_ID_XHS}")

        # Construct query image path. Assumes original_query_image_id is a filename or can be derived into one.
        query_img_filename_part = original_query_image_id #os.path.basename(original_query_image_id) 
        if '.' not in query_img_filename_part: # Add default extension if missing
            query_img_filename = f"{query_img_filename_part}.jpg"
        else:
            query_img_filename = query_img_filename_part
        query_img_path_for_jsonl = os.path.join(XHS_IMAGE_SUBDIR, query_img_filename)
        download_and_draw_box(original_query_image_url, query_img_path_for_jsonl)
        
        # CRITICAL: Placeholder for query_txt. User needs to replace this with actual query text.
        query_txt_for_jsonl = "" #f"Retrieve content related to the object bounded by the red box"
        
        query_modality = "image,text" # Default, adjust based on actual query content availability
        # Example: if not query_txt_for_jsonl: query_modality = "image"
        # Example: if not os.path.exists(actual_query_image_path_to_check): query_modality = "text"

        query_entry = {
            "qid": mbeir_qid,
            "query_txt": query_txt_for_jsonl,
            "query_img_path": query_img_path_for_jsonl,
            "box": original_query_box,
            "query_modality": query_modality,
            "query_src_content": json.dumps({"id": original_query_id}),
            "pos_cand_list": [mbeir_pos_cand_did_str] if mbeir_pos_cand_did_str else [],
            "neg_cand_list": list(set(mbeir_neg_cand_dids_list)), # Ensure unique neg DIDs
            "task_id": int(TASK_ID_XHS)
        }
        queries_jsonl_lines.append(json.dumps(query_entry))
        query_counter += 1

    cand_pool_jsonl_lines = [json.dumps(content) for content in global_mbeir_cand_contents.values()]
    if ON_SERVER:
        qrels_file_path = os.path.join(output_dir, f"qrels/test/mbeir_xhs_task{TASK_ID_XHS}_test_qrels.txt")
        queries_file_path = os.path.join(output_dir, f"query/test/mbeir_xhs_task{TASK_ID_XHS}_test.jsonl")
        cand_pool_file_path = os.path.join(output_dir, f"cand_pool/local/mbeir_xhs_task{TASK_ID_XHS}_cand_pool.jsonl")
        if TRAIN:
            qrels_file_path = os.path.join(output_dir, f"qrels/train/mbeir_xhsnote_task{TASK_ID_XHS}_train_qrels.txt")
            queries_file_path = os.path.join(output_dir, f"query/train/mbeir_xhsnote_task{TASK_ID_XHS}_train.jsonl")
            cand_pool_file_path = os.path.join(output_dir, f"cand_pool/local/mbeir_xhsnote_task{TASK_ID_XHS}_cand_pool.jsonl")
    else:
        qrels_file_path = os.path.join(output_dir, f"mbeir_xhs_task{TASK_ID_XHS}_test_qrels.txt")
        queries_file_path = os.path.join(output_dir, f"mbeir_xhs_task{TASK_ID_XHS}_test.jsonl")
        cand_pool_file_path = os.path.join(output_dir, f"mbeir_xhs_task{TASK_ID_XHS}_cand_pool.jsonl")

    with open(qrels_file_path, 'w', encoding='utf-8') as f:
        for line in qrels_lines:
            f.write(line + '\n')
    print(f"Generated qrels file: {qrels_file_path} ({len(qrels_lines)} entries)")

    with open(queries_file_path, 'w', encoding='utf-8') as f:
        for line in queries_jsonl_lines:
            f.write(line + '\n')
    print(f"Generated queries file: {queries_file_path} ({len(queries_jsonl_lines)} entries)")

    with open(cand_pool_file_path, 'w', encoding='utf-8') as f:
        for line in cand_pool_jsonl_lines:
            f.write(line + '\n')
    print(f"Generated candidate pool file: {cand_pool_file_path} ({len(cand_pool_jsonl_lines)} entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate M-BEIR evaluation files from filtered XHS samples.")
    parser.add_argument("--filtered_samples_path", type=str, required=True, 
                        help="Path to the filtered_sample.json file (output of filter_our_test1k.py).")
    parser.add_argument("--filtered_labels_path", type=str, required=True, 
                        help="Path to the filtered_labels.json file (output of filter_our_test1k.py).")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the generated M-BEIR files.")
    
    args = parser.parse_args()
    
    generate_mbeir_files(args.filtered_samples_path, args.filtered_labels_path, args.output_dir)

'''
python3 dataset/xhs_datatools/xhs_to_mbeir_format.py \
    --filtered_samples_path '/Users/liangxun/Desktop/filtered_sample.json' \
    --filtered_labels_path '/Users/liangxun/Desktop/filtered_labels.json' \
    --output_dir ./MBEIR

python3 dataset/xhs_datatools/xhs_to_mbeir_format.py \
    --filtered_samples_path dataset/xhs_datatools/sampled_20250304.json \
    --filtered_labels_path dataset/xhs_datatools/sampled_20250304_labels.json \
    --output_dir /mnt/tidal-alsh01/dataset/mmeb/M-BEIR

python3 dataset/xhs_datatools/xhs_to_mbeir_format.py \
    --filtered_samples_path /mnt/tidal-alsh01/dataset/mmeb/xhs_data/note_data/20250304/20250304.json \
    --filtered_labels_path /mnt/tidal-alsh01/dataset/mmeb/xhs_data/note_data/20250304/20250304_label.json \
    --output_dir /mnt/tidal-alsh01/dataset/mmeb/M-BEIR

python3 dataset/xhs_datatools/xhs_to_mbeir_format.py \
    --filtered_samples_path /mnt/tidal-alsh01/dataset/mmeb/xhs_data/note_data/20250304/filtered_20250304_80K.json \
    --filtered_labels_path /mnt/tidal-alsh01/dataset/mmeb/xhs_data/note_data/20250304/20250304_label.json \
    --output_dir /mnt/tidal-alsh01/dataset/mmeb/M-BEIR
'''




# import asyncio
# from collections import deque

# async def worker(name, queue, results):
#     while True:
#         item = await queue.get()
#         if item is None:  # 终止信号
#             break
            
#         # 处理任务
#         result = await async_task(item)
#         results.append(result)
#         queue.task_done()

# async def controlled_concurrency():
#     queue = asyncio.Queue(maxsize=100)
#     results = []
    
#     # 启动4个工作协程
#     workers = [asyncio.create_task(worker(f"Worker-{i}", queue, results)) 
#                for i in range(4)]
    
#     # 提交1000个任务
#     for i in range(1, 1001):
#         await queue.put(i)
        
#         # 每100个任务等待队列清空
#         if i % 100 == 0:
#             print(f"\n--- Reached {i} items, waiting ---")
#             await queue.join()  # 等待队列所有任务完成
#             print(f"Processed batch up to {i}, results count: {len(results)}")
    
#     # 发送终止信号
#     for _ in range(len(workers)):
#         await queue.put(None)
#     await asyncio.gather(*workers)

# asyncio.run(controlled_concurrency())
