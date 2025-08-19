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
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from datasets import load_dataset
import numpy as np
import ast
import pickle
import random
import glob
from io import BytesIO

# Constants for M-BEIR format
ROOT_FS = "/mnt/tidal-alsh01/dataset/mmeb/"
ON_SERVER = True
DOWNLOAD_IMAGE = False
TRAIN = False
CONCURRENCY = 100
DAM_PORT = os.environ.get("DAM_PORT", "8000")

def cal_box_score(box) -> float:
    return 1/(box[2]-box[0])/(box[3]-box[1])

def download_and_draw_box(url, filename, box=None):
    if not ON_SERVER or not DOWNLOAD_IMAGE:
        return
    if os.path.exists(filename):
        return

    response = requests.get(url, stream=True)
    if response.status_code == 403:
        size = (64, 64)
        white_img = Image.new("RGB", size, (255, 255, 255))
        white_img.save(filename)
        print(response)
        return False

    response.raise_for_status()
    img = Image.open(response.raw)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img.save(filename)
    return True

import debugpy
def setup_debugpy(local_rank):
    print(f"Debugger listening on rank {local_rank}")
    debugpy.listen(("0.0.0.0", 9999))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
# setup_debugpy(0)


@dataclass
class DocCandidate:
    """Represents a document with its associated metadata"""
    doc_image_path: str = None
    doc_image_url: str = None
    doc_box: Dict = None
    caption: str = ""


class QueryCandidate:
    """Represents a query with its positive and negative candidates"""
    def __init__(self, query_id: str, query_image_id: str = None, query_image_url: str = None, 
                 query_box: Dict = None, query_txt: str = "", 
                 candidate_items: List[DocCandidate] = None,
                 best_pos_keys: List[int] = None,
                 neg_keys: List[int] = None):
        self.query_id = query_id
        self.query_image_id = query_image_id
        self.query_image_url = query_image_url
        self.query_box = query_box
        self.query_txt = query_txt
        self.candidate_items = candidate_items or []  # All candidate items in order
        self.best_pos_keys = best_pos_keys or []      # Keys of best positive candidates
        self.neg_keys = neg_keys or []                # Keys of negative candidates

class FileExtractor(ABC):
    """Abstract base class for extracting data from different dataset formats"""
    def __init__(self, samples_path: str, labels_path: str = "", 
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 skip_samples: int = 0):
        self.samples_path = samples_path
        self.labels_path = labels_path
        self.ROOT_FS = root_fs
        self.SAMPLES_FOR_TEST_UP = samples_for_test_up
        self.SAMPLES_FOR_TEST_LOW = samples_for_test_low
        self.TOP_K_POSCAND = top_k_poscand
        self.SKIP_SAMPLES = skip_samples

    @abstractmethod
    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        """Extract queries and candidates from the dataset files"""
        pass
    def get_top_k_poscand(self, temp_pos_candidates_scores):
        if len(temp_pos_candidates_scores) <= self.TOP_K_POSCAND:
            return [x[0] for x in temp_pos_candidates_scores]
        else:
            return [x[0] for x in temp_pos_candidates_scores[:self.TOP_K_POSCAND]]
    
    def save_pil_image(self, pil_image: Image.Image, filename: str) -> bool:
        """Save PIL Image to local file"""
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if not os.path.exists(filename):
                pil_image.save(filename)
                return True
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
            return False

class XHSNoteFileExtractor(FileExtractor):
    """Extractor for XHS note dataset format"""

    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "xhs_data/note_data/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples)
        self.IMAGE_SUBDIR = image_subdir
    
    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        """Extract queries and candidates from XHS note format files"""
        samples_path = os.path.join(self.ROOT_FS, self.samples_path)
        
        with open(samples_path, 'r', encoding='utf-8') as f:
            queries_input = json.load(f)
        
        query_candidates = []
        
        for counter, (original_query_id, candidate_items_for_query) in enumerate(queries_input.items()):
            if counter < self.SKIP_SAMPLES:
                continue
            if counter >= self.SAMPLES_FOR_TEST_UP:
                break
            if counter <= self.SAMPLES_FOR_TEST_LOW:
                continue

            original_query_image_id = "_".join(candidate_items_for_query[0].get("query_image_id").split("/"))
            original_query_image_url = candidate_items_for_query[0].get("query_image_url")
            original_query_box = candidate_items_for_query[0].get("query_box")

            # Build score mapping exactly like original
            sorted_original_score_keys_for_current_query_candidates = {}
            for j, dd in enumerate(candidate_items_for_query):
                sorted_original_score_keys_for_current_query_candidates[j+1] = {
                    'img2img': dd['img2img'],
                    'box_score': cal_box_score(dd['doc_box'])
                }

            # Convert candidate_items_for_query to list of DocCandidate
            doc_candidates = []
            for item in candidate_items_for_query:
                doc_candidates.append(DocCandidate(
                    doc_image_url=item.get("doc_image_url"),
                    doc_image_path="_".join(item.get("doc_image_id", "").split("/")) + ".jpg",
                    doc_box=item.get("doc_box"),
                    caption=item.get("caption", "")
                ))
            candidate_items_for_query = doc_candidates


            if len(candidate_items_for_query) != len(sorted_original_score_keys_for_current_query_candidates):
                print(f"Warning: Mismatch in candidate list length ({len(candidate_items_for_query)}) and score key count ({len(sorted_original_score_keys_for_current_query_candidates)}) for query '{original_query_id}'. Skipping this query.")
                continue

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
            
            if not temp_pos_candidates_scores:
                continue
            elif len(temp_pos_candidates_scores) == len(candidate_items_for_query):
                continue

            # Sort positive candidates by score exactly like original
            temp_pos_candidates_scores.sort(key=lambda x: (-x[1], -x[2]))
            best_pos_original_score_key = self.get_top_k_poscand(temp_pos_candidates_scores)

            query_candidate = QueryCandidate(
                query_id=original_query_id,
                query_image_id=original_query_image_id,
                query_image_url=original_query_image_url,
                query_box=original_query_box,
                query_txt="",  # Goods dataset doesn't have query text
                candidate_items=candidate_items_for_query,
                best_pos_keys=best_pos_original_score_key,
                neg_keys=temp_neg_original_score_keys
            )
            
            query_candidates.append(query_candidate)
        
        return query_candidates

class XHSGoodsFileExtractor(FileExtractor):
    """Extractor for XHS goods dataset format"""
    
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "xhs_data/goods_data/from_20250401_to_20250407/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples)
        self.IMAGE_SUBDIR = image_subdir
    
    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        """Extract queries and candidates from XHS goods format files"""
        samples_path = os.path.join(self.ROOT_FS, self.samples_path)
        
        with open(samples_path, 'r', encoding='utf-8') as f:
            queries_input = json.load(f)
        
        query_candidates = []
        
        for counter, (original_query_id, candidate_items_for_query) in enumerate(queries_input.items()):
            if counter < self.SKIP_SAMPLES:
                continue
            # Apply sampling limits using instance variables
            if counter >= self.SAMPLES_FOR_TEST_UP:
                break
            if counter <= self.SAMPLES_FOR_TEST_LOW:
                continue

            original_query_image_id = candidate_items_for_query[0].get("query_image_path").split("/")[-1]
            original_query_image_url = candidate_items_for_query[0].get("query_image_url")
            original_query_box = candidate_items_for_query[0].get("query_box")

            # Build score mapping exactly like original
            sorted_original_score_keys_for_current_query_candidates = {}
            for j, dd in enumerate(candidate_items_for_query):
                sorted_original_score_keys_for_current_query_candidates[j+1] = {
                    'img2img': dd['img2img'],
                    'box_score': cal_box_score(dd['doc_box'])
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
                continue

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
            
            if not temp_pos_candidates_scores:
                continue
            elif len(temp_pos_candidates_scores) == len(candidate_items_for_query):
                continue

            # Sort positive candidates by score exactly like original
            temp_pos_candidates_scores.sort(key=lambda x: (-x[1], -x[2]))
            best_pos_original_score_key = self.get_top_k_poscand(temp_pos_candidates_scores)

            query_candidate = QueryCandidate(
                query_id=original_query_id,
                query_image_id=original_query_image_id,
                query_image_url=original_query_image_url,
                query_box=original_query_box,
                query_txt="",  # Goods dataset doesn't have query text
                candidate_items=candidate_items_for_query,  # ensure a list of DocCandidate
                best_pos_keys=best_pos_original_score_key,
                neg_keys=temp_neg_original_score_keys
            )
            
            query_candidates.append(query_candidate)
        
        return query_candidates


class VisMinFileExtractor(FileExtractor):
    """Extractor for Hugging Face dataset format"""
    
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "hf_data/images",
                 samples_for_test_up: int = 1000,
                 samples_for_test_low: int = 0,
                 top_k_poscand: int = 10,
                 cache_dir: Optional[str] = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
                 num_proc: int = 16,
                 mode: str = 'test',
                 skip_samples: int = 0,
                 **kwargs
                 ):
        super().__init__(samples_path, labels_path, root_fs, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples)
        self.IMAGE_SUBDIR = image_subdir
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.mode = mode

    def bbox_xywh_to_normalized_xyxy(self, boxin, image_w, image_h):
        xmin, ymin, w, h = boxin
        x1 = xmin
        y1 = ymin
        x2 = xmin + w
        y2 = ymin + h

        x1_n = x1 / image_w
        y1_n = y1 / image_h
        x2_n = x2 / image_w
        y2_n = y2 / image_h
        return [x1_n, y1_n, x2_n, y2_n]

    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        """Extract queries and candidates from Hugging Face dataset"""
        # Load dataset with specified cache_dir and num_proc
        print(f"Loading dataset from {self.samples_path}...")
        dataset = load_dataset(self.samples_path, cache_dir=self.cache_dir, num_proc=self.num_proc)
        
        # Use train split by default, or the first available split
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
        
        # Apply skip_samples if specified
        if self.SKIP_SAMPLES > 0:
            data = data.skip(self.SKIP_SAMPLES)
        
        print(f"Dataset loaded with {len(data)} samples")
        
        # Apply instance path modifications
        image_subdir_full = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR)
        os.makedirs(image_subdir_full, exist_ok=True)
        
        # State-based processing
        query_candidates = []
        temp_query_candidates = []  # 临时list存储当前query的候选项
        temp_img_save_tuples = []  # 临时存储图片保存路径和PIL对象
        current_query = None
        current_query_idx = None
        counter = 0
        
        print("Processing dataset with state-based approach...")
        for idx, item in enumerate(tqdm(data, desc="Processing dataset")):
            # # TODO
            # if idx > 5000:
            #     break

            box = item.get('bounding_boxes')
            if box == '' or box == '[]':
                box = None
            else:
                box = ast.literal_eval(box)
                
                # Handle format like [["a kite", [250, 50, 150, 200]], ["a child", [300, 300, 150, 150]]]
                if (isinstance(box, list) and len(box) > 0 and 
                    isinstance(box[0], list) and len(box[0]) == 2 and 
                    isinstance(box[0][0], str) and isinstance(box[0][1], list)):
                    # Extract only the coordinate parts from labeled boxes
                    w, h = item.get('image').size
                    coordinate_boxes = [self.bbox_xywh_to_normalized_xyxy(item[1], w, h) for item in box if len(item) == 2 and isinstance(item[1], list)]
                    if coordinate_boxes:
                        box = coordinate_boxes
                
                # Handle single nested list case
                if (isinstance(box, list) and
                    len(box) == 1 and
                    isinstance(box[0], list)):
                    box = box[0]
                    
                if (np.array(box)>1).any(): continue
            item['bounding_boxes'] = box

            # 处理box有很多的case
            if box is not None and len(box)>1 and isinstance(box[0], list):
                box = np.array(box)
                box = [min(box[:,0]).item(), min(box[:,1]).item(),max(box[:,2]).item(),max(box[:,3]).item()]
                item['bounding_boxes'] = box

                ####
                # from PIL import Image, ImageDraw
                # img = item['image']
                # W, H = img.size                          # 原始宽高
                # box_norm = (np.array(box) * [W, H, W, H]).round().astype(int).tolist()
                # draw = ImageDraw.Draw(img)
                # draw.rectangle(box_norm, outline='red', width=3)
                # img.save('out.png')
                ####
            
            # If box is empty/None, this is a query
            is_query = item["source_image_id"].strip() == ''# not box or box == [] or (isinstance(box, list) and len(box) == 4 and all(x == 0 for x in box))
            
            if is_query:
                # This is a new query - first process previous query if exists
                if current_query is not None and self.satisfy_poscand_len(temp_query_candidates):
                    # Only keep queries that have more than 1 candidate
                    if counter >= self.SAMPLES_FOR_TEST_LOW and counter < self.SAMPLES_FOR_TEST_UP:
                        self.save_images(temp_img_save_tuples, image_subdir_full)
                        query_candidates.extend(temp_query_candidates)
                    counter += 1
                
                # Start new query
                current_query = item
                current_query_idx = idx
                temp_query_candidates = []  # Reset temporary list
                temp_img_save_tuples = []
                if counter >= self.SAMPLES_FOR_TEST_UP:
                    break
                    
            else:
                # This is a candidate with box
                if current_query is not None:
                    # Create candidate and use its text to replace query's text
                    query_with_cand_text = current_query.copy()
                    query_with_cand_text['caption'] = item.get('caption', item.get('text', ''))
                    
                    # Create QueryCandidate with this single candidate
                    query_candidate, img_save_list = self._create_single_query_candidate(
                        idx, query_with_cand_text, current_query_idx, item, image_subdir_full)
                    temp_query_candidates.append(query_candidate)
                    temp_img_save_tuples.extend(img_save_list)


        if current_query is not None and self.satisfy_poscand_len(temp_query_candidates):
            if counter >= self.SAMPLES_FOR_TEST_LOW and counter < self.SAMPLES_FOR_TEST_UP:
                self.save_images(temp_img_save_tuples, image_subdir_full)
                query_candidates.extend(temp_query_candidates)
        
        print(f"Extracted {len(query_candidates)} query candidates")
        return query_candidates

    def satisfy_poscand_len(self, temp_query_candidates):
        if self.mode == 'test':
            # if test, only use those with multiple pos cands
            flag = len(temp_query_candidates) > 1
        elif self.mode == 'train':
            flag = len(temp_query_candidates) >= 1
        return flag

    def get_qimg_name(self, query_idx):
        return f"query_{query_idx}.jpg"
    def get_dimg_name(self, query_idx, rownum):
        return f"query_{query_idx}_doc_{rownum}.jpg"

    def _create_single_query_candidate(self, rownum, query_item, query_idx, cand_item, image_subdir_full):
        """Create a single QueryCandidate from query and one candidate"""
        # Add skip offset to maintain global unique IDs
        global_query_idx = query_idx + self.SKIP_SAMPLES
        global_rownum = rownum + self.SKIP_SAMPLES
        
        query_id = f"query_{global_query_idx}"
        query_image_id = self.get_qimg_name(global_query_idx)
        
        # Create doc candidate
        doc_image_id = self.get_dimg_name(global_query_idx, global_rownum)

        doc_candidate = DocCandidate(
            doc_image_url=None,
            doc_image_path=doc_image_id,
            doc_box=cand_item.get('bounding_boxes'),
            caption=None
        )
        
        return QueryCandidate(
            query_id=query_id,
            query_image_id=query_image_id,
            query_image_url=None,
            query_box=cand_item.get('bounding_boxes'),# use the same box as candidate query_item.get('bounding_boxes'),
            query_txt=query_item.get('caption', query_item.get('text', '')),
            candidate_items=[doc_candidate],
            best_pos_keys = [1]
        ), [(doc_image_id, cand_item['image']), (query_image_id, query_item['image'])]

    def save_images(self, temp_image_lists ,image_subdir_full):
        """Save images for candidates and append to query candidates"""
        for path, img in temp_image_lists:
            path = os.path.join(image_subdir_full, path)
            if not os.path.exists(path):
                self.save_pil_image(img, path)

class VisMinTrainFileExtractor(VisMinFileExtractor):
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "hf_data/images",
                 samples_for_test_up: int = 10000,
                 samples_for_test_low: int = 0,
                 top_k_poscand: int = 10,
                 cache_dir: Optional[str] = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
                 num_proc: int = 16,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, image_subdir, samples_for_test_up, samples_for_test_low, top_k_poscand, cache_dir, num_proc, mode = 'train', skip_samples=skip_samples)
    
    def get_qimg_name(self, query_idx):
        return f"{query_idx + self.SKIP_SAMPLES}.jpg"
    def get_dimg_name(self, query_idx, rownum):
        return f"{rownum + self.SKIP_SAMPLES}.jpg"
    def save_images(self, temp_image_lists ,image_subdir_full):
        pass

from dataset.datasets_dam import mask2box, counts_to_mask, lower_resolution

class DamT3FileExtractor(FileExtractor):
    """Extractor for DAM (Describe Anything Model) dataset format"""
    
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "dam_data/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 mode: str = 'draw',  # 'crop' or 'draw' mode from DAM dataset
                 cache_dir: Optional[str] = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
                 skip_samples: int = 0,
                 **kwargs):
        super().__init__(samples_path, labels_path, root_fs, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples)
        self.IMAGE_SUBDIR = image_subdir
        self.mode = mode
        self.cache_dir = cache_dir
        # DAM dataset split names
        # self.split_names = ["SAM", 'COCOStuff', 'LVIS', 'Mapillary', 'OpenImages', 'PACO']
        self.split_names = ["SAM"]

    
    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        """Extract queries and candidates from DAM dataset"""
        print(f"Loading DAM dataset from {self.samples_path}...")
        
        # Load DAM dataset
        dataset = {k: load_dataset(self.samples_path, k, num_proc=16) for k in self.split_names}
        lengths = [len(dataset[n]['train']) for n in self.split_names]
        
        print(f"Dataset loaded with splits: {dict(zip(self.split_names, lengths))}")
        
        image_subdir_full = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR)
        os.makedirs(image_subdir_full, exist_ok=True)
        
        query_candidates = []
        counter = 0
        
        print("Processing DAM dataset...")
        
        # Process each split
        for split_idx, split_name in enumerate(self.split_names):
            split_data = dataset[split_name]['train']

            if self.SKIP_SAMPLES > 0: 
                split_data = split_data.skip(self.SKIP_SAMPLES)
            
            for item_idx, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                if counter >= self.SAMPLES_FOR_TEST_UP:
                    break
                if counter < self.SAMPLES_FOR_TEST_LOW:
                    counter += 1
                    continue
                
                # Get image and annotations
                image = item['jpg']  # PIL Image
                pickle_data = pickle.loads(item['pickle'])
                
                # Create query candidates for each annotation in this item
                for anno_idx, anno in enumerate(pickle_data):
                    try:
                        # Extract annotation data
                        text = anno['caption']
                        mask = counts_to_mask(anno['mask_rle'])
                        box = mask2box(mask)
                        
                        if box is None:
                            continue
                            
                        # Filter out small boxes or extreme ratios
                        w, h = image.size
                        delta = min((box[2]-box[0])*w, (box[3]-box[1])*h)
                        wb, hb = (box[2]-box[0])*w, (box[3]-box[1])*h
                        ratio = max(hb, wb) / min(hb, wb) if min(hb, wb) > 0 else float('inf')
                        
                        if delta < 10 or ratio > 200:  # Skip small boxes or extreme ratios
                            continue
                        
                        # Process image
                        processed_image = lower_resolution(image)
                        
                        global_item_idx = item_idx + self.SKIP_SAMPLES
                        
                        query_id = f"dam_{split_name}_{global_item_idx}_{anno_idx}"
                        query_image_id = f"dam_{split_name}_{global_item_idx}.jpg"
                        doc_image_id = f"dam_{split_name}_{global_item_idx}.jpg"
                        
                        # Save images
                        query_image_path = os.path.join(image_subdir_full, query_image_id)
                        doc_image_path = os.path.join(image_subdir_full, doc_image_id)
                        
                        self.save_pil_image(processed_image, query_image_path)
                        self.save_pil_image(processed_image, doc_image_path)
                        
                        query_candidates.append(
                            self.build_query(text, query_id, query_image_id, box)
                        )
                        counter += 1
                        
                        if counter >= self.SAMPLES_FOR_TEST_UP:
                            break
                            
                    except Exception as e:
                        print(f"Error processing annotation {anno_idx} in item {item_idx}: {e}")
                        continue
                
                if counter >= self.SAMPLES_FOR_TEST_UP:
                    break
            
            if counter >= self.SAMPLES_FOR_TEST_UP:
                break
        
        print(f"Extracted {len(query_candidates)} query candidates from DAM dataset")
        return query_candidates
    
    def build_query(self, text, query_id, query_image_id, box):
        # Create doc candidate with detailed description as caption
        doc_candidate = DocCandidate(
            doc_image_url=None,
            doc_image_path=None,
            doc_box=None,
            caption=text  # Detailed description of the region
        )

        query_candidate = QueryCandidate(
            query_id=query_id,
            query_image_id=query_image_id,
            query_image_url=None,
            query_box=box,
            query_txt="",
            candidate_items=[doc_candidate],
            best_pos_keys=[1],  # Single positive candidate
            neg_keys=[]  # No negative candidates in this simple setup
        )
        return query_candidate

class DamT0FileExtractor(DamT3FileExtractor):
    """Extractor for DAM T0 t2i dataset format"""
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "dam_data/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 mode: str = 'draw',  # 'crop' or 'draw' mode from DAM dataset
                 cache_dir: Optional[str] = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
                 skip_samples: int = 0,
                 **kwargs):
        super().__init__(samples_path, labels_path, root_fs, image_subdir, samples_for_test_up, samples_for_test_low, top_k_poscand, mode, cache_dir, skip_samples)
        self.split_names = ["SAM"]
    def build_query(self, text, query_id, query_image_id, box):
        # Create doc candidate with detailed description as caption
        doc_candidate = DocCandidate(
            doc_image_url=None,
            doc_image_path=query_image_id,
            doc_box=box,
            caption=None  # Detailed description of the region
        )

        query_candidate = QueryCandidate(
            query_id=query_id,
            query_image_id=None,
            query_image_url=None,
            query_box=None,
            query_txt=text,
            candidate_items=[doc_candidate],
            best_pos_keys=[1],  # Single positive candidate
            neg_keys=[]  # No negative candidates in this simple setup
        )
        return query_candidate

class FgclipT3FileExtractor(DamT3FileExtractor):
    """Extractor for FGCLIP dataset format, similar to DamT3FileExtractor"""
    
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "fgclip_data/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 cache_dir: Optional[str] = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
                 use_dam: bool = False,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, image_subdir, samples_for_test_up, samples_for_test_low, top_k_poscand, 'draw', cache_dir, skip_samples)
        self.use_dam = use_dam
        if use_dam:
            from dataset.xhs_datatools.dam_client import SimpleDAMClient
            self.dam_client = SimpleDAMClient(
                f"http://localhost:{DAM_PORT}",
                "describe_anything_model"
            )
    def lower_resolution(self, img: Image.Image):
        """Lower image resolution if too large"""
        h, w = img.size
        if h > 1000 and w > 1000:
            return img.resize((h//2, w//2))
        else:
            return img

    def load_dataset(self):
        parquet_files = glob.glob(os.path.join(self.samples_path, '*.parquet'))
        ds = load_dataset('parquet', num_proc=16, data_files=parquet_files)['train']
        if self.SKIP_SAMPLES > 0:
            ds = ds.skip(self.SKIP_SAMPLES)
        return ds

    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        dataset = self.load_dataset()
        image_subdir_full = os.path.join(self.ROOT_FS, self.IMAGE_SUBDIR)
        os.makedirs(image_subdir_full, exist_ok=True)
        
        query_candidates = []
        counter = 0
        
        print("Processing FGCLIP dataset...")
        
        # Process dataset items
        for idx, item in enumerate(tqdm(dataset, desc="Processing FGCLIP")):
            if counter >= self.SAMPLES_FOR_TEST_UP:
                break
            if counter < self.SAMPLES_FOR_TEST_LOW:
                counter += 1
                continue
            
            bbox_infos = item.get("bbox_info", [])
            if len(bbox_infos) < 1:
                continue
            else:
                lens = [len(t['long_expr']) for t in bbox_infos]
                # Get indices of 2 longest expressions
                sorted_indices = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)
                valid_indices = [i for i in sorted_indices if lens[i] > 0]
                if len(valid_indices) < 2:
                    continue
                bbox_infos = [bbox_infos[i] for i in valid_indices[:2]]

            image = Image.open(BytesIO(item['image'])).convert("RGB")
            # Lower resolution if needed
            image = self.lower_resolution(image)
            w, h = image.size
            if w < 10 or h < 10:
                continue

            for bbox_idx, selected_bbox in zip(valid_indices, bbox_infos):
                bbox = selected_bbox.get("bbox", [0, 0, 1., 1, -1])
                bbox = bbox[:4]

                bbox_text = selected_bbox.get("long_expr", "")
                if self.use_dam:
                    bbox_text = self.dam_client.describe_region(
                        image=image,
                        box=bbox,
                        query="\nDescribe the region.",
                        stream=False
                    )
                
                global_idx = idx + self.SKIP_SAMPLES
                
                query_id = f"fgclip_{global_idx}_{bbox_idx}"
                query_image_id = f"fgclip_{global_idx}.jpg"
                
                # Save processed image
                query_image_path = os.path.join(image_subdir_full, query_image_id)
                self.save_pil_image(image, query_image_path)
                
                query_candidates.append(
                    self.build_query(bbox_text, query_id, query_image_id, bbox)
                )
                counter += 1
                
                if counter >= self.SAMPLES_FOR_TEST_UP:
                    break
            
            if counter >= self.SAMPLES_FOR_TEST_UP:
                break
        
        print(f"Extracted {len(query_candidates)} query candidates from FGCLIP dataset")
        return query_candidates

class FgclipT0FileExtractor(FgclipT3FileExtractor):
    """Extractor for FGCLIP dataset format, similar to DamT3FileExtractor"""
    
    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "fgclip_data/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 cache_dir: Optional[str] = "/mnt/tidal-alsh01/dataset/mmeb/.cache/huggingface/dataset",
                 use_dam: bool = False,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, image_subdir, samples_for_test_up, samples_for_test_low, top_k_poscand, cache_dir, use_dam, skip_samples)

    def build_query(self, text, query_id, query_image_id, box):
        # Create doc candidate with detailed description as caption
        doc_candidate = DocCandidate(
            doc_image_url=None,
            doc_image_path=query_image_id,
            doc_box=box,
            caption=None  # Detailed description of the region
        )

        query_candidate = QueryCandidate(
            query_id=query_id,
            query_image_id=None,
            query_image_url=None,
            query_box=None,
            query_txt=text,
            candidate_items=[doc_candidate],
            best_pos_keys=[1],  # Single positive candidate
            neg_keys=[]  # No negative candidates in this simple setup
        )
        return query_candidate

class ImgDiffFileExtractor(FileExtractor):
    """Extractor for ImgDiff dataset format (sharegpt format)"""

    def __init__(self, samples_path: str, labels_path: str = "",
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "Img-Diff/images",
                 samples_for_test_up: int = 5000,
                 samples_for_test_low: int = 1000,
                 top_k_poscand: int = 10,
                 skip_samples: int = 0, **kwargs):
        super().__init__(samples_path, labels_path, root_fs, samples_for_test_up, samples_for_test_low, top_k_poscand, skip_samples)
        self.IMAGE_SUBDIR = image_subdir

    def extract_queries_and_candidates(self) -> List[QueryCandidate]:
        data_path = self.samples_path
        if not data_path.startswith("/"):
            data_path = os.path.join(self.ROOT_FS, self.samples_path)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if self.SKIP_SAMPLES > 0:
                data = data[self.SKIP_SAMPLES:]
        
        print(f"Loaded {len(data)} samples from Imgdiff dataset")
        
        query_candidates = []
        counter = 0
        
        for idx, item in enumerate(tqdm(data, desc="Processing Imgdiff")):
            if counter >= self.SAMPLES_FOR_TEST_UP:
                break
            if counter < self.SAMPLES_FOR_TEST_LOW:
                counter += 1
                continue

            conversations = item.get("conversations", [])
            bbox = ast.literal_eval(item.get("bbox"))
            # captions1 = item.get("captions1", "")
            # captions2 = item.get("captions2", "")
            path = item.get("path", "")
            caption = conversations[1].get("value")

            # Add skip offset for global unique IDs
            global_idx = idx + self.SKIP_SAMPLES
            query_id = f"imgdiff_{global_idx}"

            doc_candidate = DocCandidate(
                doc_image_url=None,
                doc_image_path=path+"_1.jpg",
                doc_box=bbox,
                caption=""
            )
            
            query_candidate = QueryCandidate(
                query_id=query_id,
                query_image_id=path+"_0.jpg",
                query_image_url=None,
                query_box=bbox,
                query_txt=caption,
                candidate_items=[doc_candidate],
                best_pos_keys=[1],
                neg_keys=[]
            )
            
            query_candidates.append(query_candidate)
            counter += 1

        print(f"Extracted {len(query_candidates)} query candidates from ImgDiff dataset")
        return query_candidates

class MBeirConverter:
    """Converts query candidates to M-BEIR format and writes to files"""
    
    def __init__(self, output_dir: str, 
                 dataset_id: str,
                 task_id: str,
                 dsname: str,
                 root_fs: str = "/mnt/tidal-alsh01/dataset/mmeb/",
                 image_subdir: str = "xhs_data/goods_data/from_20250401_to_20250407/images"):
        self.DATASET_ID = dataset_id
        self.TASK_ID = task_id
        self.ROOT_FS = root_fs
        self.IMAGE_SUBDIR = os.path.join(root_fs, image_subdir)
        self.output_dir = os.path.join(root_fs, output_dir)
        self.dsname = dsname
        
        # Global mappings for M-BEIR format
        self.global_orig_cand_to_mbeir_did = {}
        self.global_mbeir_cand_contents = {}
        self.query_counter = 1
        self.cand_counter = 1
        
        # Output collections
        self.qrels_lines = []
        self.queries_jsonl_lines = []
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def convert_query_candidate(self, query_candidate: QueryCandidate) -> bool:
        """Convert a single QueryCandidate to M-BEIR format"""
        mbeir_qid = f"{self.DATASET_ID}:{self.query_counter}"
        mbeir_pos_cand_dids_list = []
        mbeir_neg_cand_dids_list = []
        
        # Process candidates exactly like original code
        # Iterate through candidate_items_for_query using the sorted_original_score_keys for indexing
        for i, cand_item_dict in enumerate(query_candidate.candidate_items):
            original_score_key_for_item = i + 1  # This matches the original enumeration (j+1)
            
            original_cand_img_url = cand_item_dict.doc_image_url
            original_cand_caption = cand_item_dict.caption
            original_cand_img_path = cand_item_dict.doc_image_path
            original_box = cand_item_dict.doc_box

            unique_cand_key_part = ""
            if original_cand_img_path:
                unique_cand_key_part = os.path.basename(original_cand_img_path) + str(original_box)
            elif original_cand_caption:
                unique_cand_key_part = original_cand_caption[:50]
            else:
                unique_cand_key_part = f"unidentified_cand_oqid_{query_candidate.query_id}_osk_{original_score_key_for_item}"
            
            original_cand_identifier = unique_cand_key_part
            add_to_global_pool = True
            
            if original_cand_identifier not in self.global_orig_cand_to_mbeir_did:
                mbeir_cand_did = f"{self.DATASET_ID}:{self.cand_counter}"
                self.global_orig_cand_to_mbeir_did[original_cand_identifier] = mbeir_cand_did
                
                cand_img_path_for_pool = None
                if original_cand_img_path:
                    # Use relative path from IMAGE_SUBDIR
                    cand_img_path_for_pool = os.path.join(self.IMAGE_SUBDIR, original_cand_img_path)
                    download_and_draw_box(original_cand_img_url, cand_img_path_for_pool)

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
                    "img_path": original_cand_img_path,
                    "modality": modality,
                    "did": mbeir_cand_did,
                    "box": original_box,
                    "src_content": json.dumps({
                        "original_id": original_cand_identifier, 
                        "original_query_id_context": query_candidate.query_id, 
                        "original_score_key_for_query": original_score_key_for_item
                    }),
                    "txt": cand_txt_for_pool
                }
                add_to_global_pool = True
            else:
                add_to_global_pool = False
                mbeir_cand_did = self.global_orig_cand_to_mbeir_did[original_cand_identifier]

            # Add to appropriate lists exactly like original
            if original_score_key_for_item in query_candidate.best_pos_keys:
                mbeir_pos_cand_dids_list.append(mbeir_cand_did)
                if add_to_global_pool:
                    self.global_mbeir_cand_contents[mbeir_cand_did] = cand_content_for_pool
                    self.cand_counter += 1
            elif original_score_key_for_item in query_candidate.neg_keys:
                mbeir_neg_cand_dids_list.append(mbeir_cand_did)
                if add_to_global_pool:
                    self.global_mbeir_cand_contents[mbeir_cand_did] = cand_content_for_pool
                    self.cand_counter += 1

        # Add to qrels exactly like original - multiple positive candidates
        for mbeir_pos_cand_did_str in mbeir_pos_cand_dids_list:
            self.qrels_lines.append(f"{mbeir_qid} 0 {mbeir_pos_cand_did_str} 1 {self.TASK_ID}")

        query_modality = "unknown"
        has_image = bool(query_candidate.query_image_id)
        has_text = bool(query_candidate.query_txt)
        if has_image and has_text:
            query_modality = "image,text"
        elif has_image:
            query_modality = "image"
        elif has_text:
            query_modality = "text"

        if has_image:
            query_img_filename_part = query_candidate.query_image_id
            if '.' not in query_img_filename_part:
                query_img_filename = f"{query_img_filename_part}.jpg"
            else:
                query_img_filename = query_img_filename_part
            query_img_path_for_jsonl = os.path.join(self.IMAGE_SUBDIR, query_img_filename)
            download_and_draw_box(query_candidate.query_image_url, query_img_path_for_jsonl)
        else:
            query_img_filename = ''

        query_entry = {
            "qid": mbeir_qid,
            "query_txt": query_candidate.query_txt,
            "query_img_path": query_img_filename,
            "box": query_candidate.query_box,
            "query_modality": query_modality,
            "query_src_content": json.dumps({"id": query_candidate.query_id}),
            "pos_cand_list": mbeir_pos_cand_dids_list,
            "neg_cand_list": list(set(mbeir_neg_cand_dids_list)),
            "task_id": int(self.TASK_ID)
        }
        
        self.queries_jsonl_lines.append(json.dumps(query_entry))
        self.query_counter += 1
        
        return True
    
    def write_output_files(self):
        if ON_SERVER:
            qrels_file_path = os.path.join(self.output_dir, f"qrels/test/mbeir_{self.dsname}_task{self.TASK_ID}_test_qrels.txt")
            queries_file_path = os.path.join(self.output_dir, f"query/test/mbeir_{self.dsname}_task{self.TASK_ID}_test.jsonl")
            cand_pool_file_path = os.path.join(self.output_dir, f"cand_pool/test/mbeir_{self.dsname}_task{self.TASK_ID}_cand_pool.jsonl")
            if TRAIN:
                qrels_file_path = os.path.join(self.output_dir, f"qrels/train/mbeir_{self.dsname}_task{self.TASK_ID}_train_qrels.txt")
                queries_file_path = os.path.join(self.output_dir, f"query/train/mbeir_{self.dsname}_task{self.TASK_ID}_train.jsonl")
                cand_pool_file_path = os.path.join(self.output_dir, f"cand_pool/local/mbeir_{self.dsname}_task{self.TASK_ID}_cand_pool.jsonl")
        else:
            qrels_file_path = os.path.join(self.output_dir, f"mbeir_{self.dsname}_task{self.TASK_ID}_test_qrels.txt")
            queries_file_path = os.path.join(self.output_dir, f"mbeir_{self.dsname}_task{self.TASK_ID}_test.jsonl")
            cand_pool_file_path = os.path.join(self.output_dir, f"mbeir_{self.dsname}_task{self.TASK_ID}_cand_pool.jsonl")

        # Create directories if needed
        os.makedirs(os.path.dirname(qrels_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(queries_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(cand_pool_file_path), exist_ok=True)
        
        # Write files exactly like original
        with open(qrels_file_path, 'w', encoding='utf-8') as f:
            for line in self.qrels_lines:
                f.write(line + '\n')
        print(f"Generated qrels file: {qrels_file_path} ({len(self.qrels_lines)} entries)")

        with open(queries_file_path, 'w', encoding='utf-8') as f:
            for line in self.queries_jsonl_lines:
                f.write(line + '\n')
        print(f"Generated queries file: {queries_file_path} ({len(self.queries_jsonl_lines)} entries)")

        cand_pool_jsonl_lines = [json.dumps(content) for content in self.global_mbeir_cand_contents.values()]
        with open(cand_pool_file_path, 'w', encoding='utf-8') as f:
            for line in cand_pool_jsonl_lines:
                f.write(line + '\n')
        print(f"Generated candidate pool file: {cand_pool_file_path} ({len(cand_pool_jsonl_lines)} entries)")

def generate_mbeir_files(filtered_samples_path, filtered_labels_path, output_dir, 
        dataset_id,
        task_id, 
        image_subdir,
        dsname,
        cls = XHSGoodsFileExtractor,
        root_fs=ROOT_FS,
        samples_for_test_up=2000, samples_for_test_low=0, top_k_poscand=10, use_dam=False,
        skip_samples=0
    ):
    extractor = cls(
        filtered_samples_path, 
        filtered_labels_path,
        root_fs=root_fs,
        image_subdir=image_subdir,
        samples_for_test_up=samples_for_test_up,
        samples_for_test_low=samples_for_test_low,
        top_k_poscand=top_k_poscand,
        use_dam=use_dam,
        skip_samples=skip_samples
    )
    converter = MBeirConverter(
        output_dir,
        dsname=dsname,
        dataset_id=dataset_id,
        task_id=task_id,
        root_fs=root_fs,
        image_subdir=image_subdir
    )
    
    query_candidates = extractor.extract_queries_and_candidates()
    for query_candidate in query_candidates:
        converter.convert_query_candidate(query_candidate)
    
    converter.write_output_files()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Generate M-BEIR evaluation files from filtered XHS samples.")
    # parser.add_argument("--filtered_samples_path", type=str, required=True, 
    #                     help="Path to the filtered_sample.json file (output of filter_our_test1k.py).")
    # parser.add_argument("--filtered_labels_path", type=str, default="", 
    #                     help="Path to the filtered_labels.json file (output of filter_our_test1k.py).")
    # parser.add_argument("--output_dir", type=str, required=True, 
    #                     help="Directory to save the generated M-BEIR files.")
    
    # args = parser.parse_args()

    # xhs goods
    generate_mbeir_files(
        # args.filtered_samples_path, args.filtered_labels_path, args.output_dir,
        "xhs_data/goods_data/from_20250401_to_20250407/from_20250401_to_20250407.json",
        "",
        "M-BEIR",
        dsname="xhsgood",
        cls = XHSGoodsFileExtractor,
        dataset_id="10", task_id="4",
        root_fs=ROOT_FS,
        image_subdir="xhs_data/goods_data/from_20250401_to_20250407/images",
        samples_for_test_up=5000, samples_for_test_low=1000, top_k_poscand=10,
        skip_samples=0
    )
    # xhs notes
    generate_mbeir_files(
        "xhs_data/note_data/20250304/filtered_20250304_80K.json",
        "xhs_data/note_data/20250304/20250304_label.json",
        "M-BEIR",
        dsname="xhsnote",
        cls = XHSNoteFileExtractor,
        dataset_id="11", task_id="4",
        root_fs=ROOT_FS,
        image_subdir="xhs_data/note_data/20250304/images",
        samples_for_test_up=1000, samples_for_test_low=0, top_k_poscand=6,
        skip_samples=0
    )

    # VisMin dataset
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/vismin/",
        "",
        "M-BEIR",
        dsname="vismin",
        cls = VisMinFileExtractor,
        dataset_id="12", task_id="6",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/vismin_images",
        samples_for_test_up=1394, samples_for_test_low=0, top_k_poscand=6,
        skip_samples=0
    )

    # DAM dataset task3
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data",  # Path to DAM dataset
        "",
        "M-BEIR",
        dsname="dam",
        cls = DamT3FileExtractor,
        dataset_id="13", task_id="3",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/dam_images",
        samples_for_test_up=1500, samples_for_test_low=0, top_k_poscand=1,
        skip_samples=0
    )

    # DAM dataset task0
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data",  # Path to DAM dataset
        "",
        "M-BEIR", 
        dsname="dam",
        cls = DamT0FileExtractor,
        dataset_id="13", task_id="0",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/dam_images",
        samples_for_test_up=1500, samples_for_test_low=0, top_k_poscand=1,
        skip_samples=0 # BUG 应该截取 1500之后的一部分尝试的，这个东西互为镜像啊
    )

    # FGCLIP dataset task3
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/fg-clip",  # Path to FGCLIP dataset
        "",
        "M-BEIR",
        dsname="fgclip",
        cls = FgclipT3FileExtractor,
        dataset_id="14", task_id="3",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/fgclip_images",
        samples_for_test_up=1500, samples_for_test_low=0, top_k_poscand=1,
        use_dam=True,
        skip_samples=0
    )

    # FGCLIP dataset task0
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/fg-clip",
        "",
        "M-BEIR",
        dsname="fgclip",
        cls = FgclipT0FileExtractor,
        dataset_id="14", task_id="0",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/fgclip_images",
        samples_for_test_up=1500, samples_for_test_low=0, top_k_poscand=1,
        use_dam=True,
        skip_samples=5000
    )
    # VisMin dataset Train
    TRAIN = True
    # 原来的代码是hard-code取前5k构造test set, train set应该从5k开始
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/vismin/",
        "",
        "M-BEIR",
        dsname="vismin",
        cls = VisMinTrainFileExtractor,
        dataset_id="12", task_id="6",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/vismin_images",
        samples_for_test_up=150000, samples_for_test_low=0, top_k_poscand=6,
        skip_samples=5000
    )
    TRAIN = False

    # Img-Diff test
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/Img-Diff/img_diff_object_replacement.json",
        "",
        "M-BEIR",
        dsname="imgdiff",
        cls = ImgDiffFileExtractor,
        dataset_id="15", task_id="6",
        root_fs=ROOT_FS,
        image_subdir="Img-Diff/filtered_new_edit_data",
        samples_for_test_up=1300, samples_for_test_low=0, top_k_poscand=6,
        skip_samples=0
    )
    # Img-Diff train
    TRAIN = True
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/Img-Diff/img_diff_object_replacement.json",
        "",
        "M-BEIR",
        dsname="imgdiff",
        cls = ImgDiffFileExtractor,
        dataset_id="15", task_id="6",
        root_fs=ROOT_FS,
        image_subdir="Img-Diff/filtered_new_edit_data",
        samples_for_test_up=30000, samples_for_test_low=0, top_k_poscand=6,
        skip_samples=5000
    )
    TRAIN = False

    # xhs notes train
    TRAIN=True
    generate_mbeir_files(
        "xhs_data/note_data/20250304/filtered_20250304_80K.json",
        "xhs_data/note_data/20250304/20250304_label.json",
        "M-BEIR",
        dsname="xhsnote",
        cls = XHSNoteFileExtractor,
        dataset_id="11", task_id="4",
        root_fs=ROOT_FS,
        image_subdir="xhs_data/note_data/20250304/images",
        samples_for_test_up=70000, samples_for_test_low=0, top_k_poscand=6,
        skip_samples=1000
    )
    TRAIN = False

    # DAM dataset task3
    TRAIN=True
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data",  # Path to DAM dataset
        "",
        "M-BEIR",
        dsname="dam",
        cls = DamT3FileExtractor,
        dataset_id="13", task_id="3",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/dam_images",
        samples_for_test_up=30000, samples_for_test_low=0, top_k_poscand=1,
        skip_samples=10000
    )
    TRAIN = False

    # DAM dataset task0
    TRAIN=True
    generate_mbeir_files(
        "/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data",
        "",
        "M-BEIR", 
        dsname="dam",
        cls = DamT0FileExtractor,
        dataset_id="13", task_id="0",
        root_fs=ROOT_FS,
        image_subdir="M-BEIR/mbeir_images/dam_images",
        samples_for_test_up=30000, samples_for_test_low=0, top_k_poscand=1,
        skip_samples=40000
    )
    TRAIN = False
'''
python3 dataset/xhs_datatools/xhs_to_mbeir_format.py
'''