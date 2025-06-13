import json
from transformers import AutoProcessor
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
import argparse
from dataset.datasets_mbeir import QueryDataset, CandidateDataset
from collators.mbeir_eval import MbeirQueryDataCollator, MbeirCandidateDataCollator
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from tqdm import tqdm

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents

    # Get the set of indices for the top k retrieved documents
    top_k_retrieved_indices_set = set(retrieved_indices[:k])

    # Convert the relevant documents to a set
    relevant_docs_set = set(relevant_docs)

    # Check if there is an intersection between relevant docs and top k retrieved docs
    # If there is, we return 1, indicating successful retrieval; otherwise, we return 0
    result = []
    for x in relevant_docs_set:
        filtered_top_k = top_k_retrieved_indices_set - (relevant_docs_set - {x})
        if x in filtered_top_k:
            result.append(1.0)
        else:
            result.append(0.0)
    return result

def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id

    import numpy as np 

    save_dir_name = "./LamRA_Ret_eval_results/note-1k-1top-1pos-filter"
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_name = args.qrels_path.split('/')[-1].replace('_qrels.txt', '')
    model_name = args.model_id.split('/')[-1]
    save_name = f"{save_name}_{model_name}"
    with open(f"{save_dir_name}/{save_name}_query_names.json", 'r') as f:
        query_names = json.load(f)
    with open(f"{save_dir_name}/{save_name}_cand_names.json", 'r') as f:
        cand_names = np.array(json.load(f))


    qrel, qid_to_taskid = load_qrel(args.qrels_path)

    k_lists = [1, 5, 10, 50]
    res = {}

    for k in k_lists:
        res[f'recall_{k}'] = []

    for ind, query_name in enumerate(tqdm(query_names)):
        relevant_docs = qrel[query_name]
        retrieved_indices_for_qid = cand_names[ind]
        for k in k_lists:
            recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
            res[f'recall_{k}'].extend(recall_at_k)

    for k in k_lists:
        print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

    model_name = model_id.split('/')[-1]
    with open(f"{save_dir_name}/{model_name}_results.txt", 'a') as f:
        f.write(args.qrels_path + '\n')
        for k in k_lists:
            f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}" + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_data_path', type=str)
    parser.add_argument('--cand_pool_path', type=str)
    parser.add_argument('--instructions_path', type=str)
    parser.add_argument('--qrels_path', type=str)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--query_cand_pool_path', type=str)
    parser.add_argument('--image_path_prefix', type=str)

    args = parser.parse_args()
    eval(args)