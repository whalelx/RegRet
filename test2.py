from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch

# ---- 配置 ----
model_path = "/mnt/tidalfs-hssh01/dataset/mmeb/Qwen3-VL-2B-Instruct"  # 改成你的 Qwen3VL 路径
image_path = "./demo.jpeg"
PROMPT_TEXT = "Describe the image."

# ---- 加载模型 ----
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
model.eval()

# ---- 构建消息 ----
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": PROMPT_TEXT},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "A human and a dog"},
        ],
    }
]

# ---- 处理输入 ----
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  # 末尾加上 <|im_start|>assistant\n
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    do_resize=True,
    return_tensors="pt",
)

device = next(model.parameters()).device
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("=== Input shapes ===")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# ---- 单次 forward ----
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=False)

logits = outputs.logits  # [1, seq_len, vocab_size]
print(f"\n=== Logits shape: {logits.shape} ===")
print(f"\n=== Loss: {outputs.loss} ===")

# ---- 解码每个位置的预测 token ----
pred_ids = logits.argmax(dim=-1)  # [1, seq_len]

# 正确对齐：logits[t] -> 预测 input_ids[t+1]
shifted_pred_ids  = pred_ids[:, :-1]          # [1, seq_len-1]
shifted_input_ids = inputs["input_ids"][:, 1:] # [1, seq_len-1]

token_acc = (shifted_pred_ids == shifted_input_ids).float().mean().item()
print(f"Teacher-forcing token accuracy: {token_acc:.4f}")

# 最后一个位置的预测 = assistant 第一个 token
next_token_id = pred_ids[:, -1:]
print("\n=== Next token (assistant 第一个 token) ===")
print(processor.batch_decode(inputs["input_ids"], skip_special_tokens=False))
print(processor.batch_decode(pred_ids, skip_special_tokens=False))
