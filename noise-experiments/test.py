from transformers import AutoProcessor, Qwen2VLForConditionalGeneration,AutoTokenizer
# from qwen_vl_utils import process_vision_info
from collators.qwen2_vl_2b import Qwen2VL2BDataCollator
import torch
messages = [
    [{"role": "user", "content": [
        {"type": "image", "image": "./demo.jpeg"},
        {"type": "text", "text": "Describe the image."}
    ]}],
]

model_path = "/mnt/tidalfs-hssh01/dataset/mmeb/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

collactor = Qwen2VL2BDataCollator(
    tokenizer=tokenizer,
    processor=processor
)
inputs=collactor(messages)

# ---- 方式一：generate（自回归生成，推荐用于生成文本）----
gen_ids = model.generate(**inputs, max_new_tokens=256)
new_ids = gen_ids[:, inputs['input_ids'].shape[1]:]  # 只取新生成的部分
print(processor.batch_decode(new_ids, skip_special_tokens=True))

# ---- 方式二：forward（提取embedding，用于检索/Ret模型）----
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=False)

# logits解码（next-token预测，不是自回归生成）
logits = outputs.logits
pred_ids = logits.argmax(dim=-1)
print(processor.batch_decode(pred_ids, skip_special_tokens=True))