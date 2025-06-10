import json
import random

input_file = '/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/union_train/mbeir_union_up_train.jsonl'
output_file = '/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/union_train/mbeir_language_train200k.jsonl'

filtered_lines = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'task_id' in data and data['task_id'] in [3]:
            filtered_lines.append(data)


total_filtered = len(filtered_lines)
print(f'含 task_id = 3 的样本总数: {total_filtered}')

# 随机shuffle后取前200000条
random.shuffle(filtered_lines)
sample_size = min(200_000, total_filtered)
sampled_lines = filtered_lines[:sample_size]

# 写入新的jsonl
with open(output_file, 'w', encoding='utf-8') as f_out:
    for item in sampled_lines:
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'已将 {sample_size} 条样本写入 {output_file}')
