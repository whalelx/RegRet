import json
import os

def read_jsonl(file_path):
    """读取JSONL文件并返回列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data, file_path):
    """将数据写入JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_qrels(file_path):
    """读取qrels文件并返回列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

def write_qrels(data, file_path):
    """将qrels数据写入文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

def get_max_id(data, key_prefix="10:"):
    """获取指定前缀的ID的最大值"""
    max_id = 0
    for item in data:
        if "qid" in item and item["qid"].startswith(key_prefix):
            id_num = int(item["qid"].split(":")[1])
            max_id = max(max_id, id_num)
        elif "did" in item and item["did"].startswith(key_prefix):
            id_num = int(item["did"].split(":")[1])
            max_id = max(max_id, id_num)
    return max_id

def update_qid_did(item, qid_offset, did_offset, key_prefix="10:"):
    """更新数据项中的qid、did以及相关列表"""
    new_item = item.copy()
    
    # 更新qid
    if "qid" in new_item and new_item["qid"].startswith(key_prefix):
        qid_parts = new_item["qid"].split(":")
        new_qid = f"{qid_parts[0]}:{int(qid_parts[1]) + qid_offset}"
        new_item["qid"] = new_qid
    
    # 更新did
    if "did" in new_item and new_item["did"].startswith(key_prefix):
        did_parts = new_item["did"].split(":")
        new_did = f"{did_parts[0]}:{int(did_parts[1]) + did_offset}"
        new_item["did"] = new_did
    
    # 更新pos_cand_list
    if "pos_cand_list" in new_item:
        new_pos_list = []
        for cand_id in new_item["pos_cand_list"]:
            if cand_id.startswith(key_prefix):
                parts = cand_id.split(":")
                new_cand_id = f"{parts[0]}:{int(parts[1]) + did_offset}"
                new_pos_list.append(new_cand_id)
            else:
                new_pos_list.append(cand_id)
        new_item["pos_cand_list"] = new_pos_list
    
    # 更新neg_cand_list
    if "neg_cand_list" in new_item:
        new_neg_list = []
        for cand_id in new_item["neg_cand_list"]:
            if cand_id.startswith(key_prefix):
                parts = cand_id.split(":")
                new_cand_id = f"{parts[0]}:{int(parts[1]) + did_offset}"
                new_neg_list.append(new_cand_id)
            else:
                new_neg_list.append(cand_id)
        new_item["neg_cand_list"] = new_neg_list
    
    return new_item

def update_qrels_line(line, qid_offset, did_offset, key_prefix="10:"):
    """更新qrels文件中的行"""
    parts = line.split()
    if len(parts) >= 3:
        if parts[0].startswith(key_prefix):
            qid_parts = parts[0].split(":")
            parts[0] = f"{qid_parts[0]}:{int(qid_parts[1]) + qid_offset}"
        
        if parts[2].startswith(key_prefix):
            did_parts = parts[2].split(":")
            parts[2] = f"{did_parts[0]}:{int(did_parts[1]) + did_offset}"
    
    return " ".join(parts)

def merge_files(
        query1, cand1,qrels1,
        query2, cand2,qrels2,
        outquery, outcand,outqrels,
        key_prefix="10:"):
    """合并两组文件"""
    # 读取第一组文件
    cand_pool1 = read_jsonl(cand1)
    query1 = read_jsonl(query1)
    qrels1 = read_qrels(qrels1)
    
    # 读取第二组文件
    cand_pool2 = read_jsonl(cand2)
    query2 = read_jsonl(query2)
    qrels2 = read_qrels(qrels2)

    # 获取第一组文件中的最大ID
    max_qid = get_max_id(query1, key_prefix)
    max_did = get_max_id(cand_pool1, key_prefix)
    
    print(f"First file max qid: {max_qid}, max did: {max_did}")
    
    # 更新第二组文件中的ID
    updated_cand_pool2 = [update_qid_did(item, max_qid, max_did, key_prefix) for item in cand_pool2]
    updated_query2 = [update_qid_did(item, max_qid, max_did, key_prefix) for item in query2]
    updated_qrels2 = [update_qrels_line(line, max_qid, max_did, key_prefix) for line in qrels2]
    
    # 合并数据
    merged_cand_pool = cand_pool1 + updated_cand_pool2
    merged_query = query1 + updated_query2
    merged_qrels = qrels1 + updated_qrels2
    
    # 写入合并后的文件
    write_jsonl(merged_cand_pool, outcand)
    write_jsonl(merged_query, outquery)
    write_qrels(merged_qrels, outqrels)

    print(f"Merged files saved with prefix: \n{outqrels}\n{outquery}\n{outcand}")

# 使用示例
if __name__ == "__main__":

    cand2 ="/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/cand_pool/test/mbeir_xhsgoodmoreneg2_task7_cand_pool.jsonl"
    qrels2 ="/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/qrels/test/mbeir_xhsgoodmoreneg2_task7_test_qrels.txt"
    query2 ="/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/test/mbeir_xhsgoodmoreneg2_task7_test.jsonl"


    cand1 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/cand_pool/test/mbeir_xhsgoodmoreneg_task7_cand_pool.jsonl"
    qrels1 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/qrels/test/mbeir_xhsgoodmoreneg_task7_test_qrels.txt"
    query1 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/test/mbeir_xhsgoodmoreneg_task7_test.jsonl"


    outquery = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/test/mbeir_xhsgood_task7_merge_test.jsonl"
    outcand = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/cand_pool/test/mbeir_xhsgoodmoreneg_task7_merge_cand_pool.jsonl"
    outqrels = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/qrels/test/mbeir_xhsgoodmoreneg_task7_merge_test_qrels.txt"

    #######################

    cand1 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/cand_pool/local/mbeir_xhsgoodtrain_task7_cand_pool.jsonl"
    qrels1 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/qrels/train/mbeir_xhsgoodtrain_task7_train_qrels.txt"
    query1 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/train/mbeir_xhsgoodtrain_task7_train.jsonl"

    cand2 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/cand_pool/local/mbeir_xhsgoodtrainv4_task7_cand_pool.jsonl"
    qrels2 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/qrels/train/mbeir_xhsgoodtrainv4_task7_train_qrels.txt"
    query2 = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/train/mbeir_xhsgoodtrainv4_task7_train.jsonl"

    outquery = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/train/mbeir_xhsgood_task7_merge_train.jsonl"
    outcand = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/cand_pool/local/mbeir_xhsgood_task7_merge_cand_pool.jsonl"
    outqrels = "/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/qrels/train/mbeir_xhsgood_task7_merge_train_qrels.txt"

    key_prefix = "10:"  # 根据实际ID前缀调整

    merge_files(
        query1, cand1,qrels1,
        query2, cand2,qrels2,
        outquery, outcand,outqrels,
        key_prefix
    )


# find "$(pwd)" -type d -name mbeir_images -prune -o -type f -name '*moreneg_*' -print