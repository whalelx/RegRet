import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "./")
sys.path.append(module_path)

import torch
import torch.nn.functional as F
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from loaders.processor import LemuirProcessor
from collators.qwen2_vision_process import process_vision_info_with_focal


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
MODEL_ID = "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"   # 可替换为你本地微调后的模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ──────────────────────────────────────────────
# 加载模型与处理器
# ──────────────────────────────────────────────
model = Qwen2VLRetForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", 
    low_cpu_mem_usage=True,
).to(DEVICE)

processor = LemuirProcessor.from_pretrained("./checkpoints/LamRA-Ret")
tokenizer = processor.tokenizer
tokenizer.padding_side = "left"


def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    if len(emb_tokens) == num_new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids


add_embed_token(tokenizer, model)
model.eval()


# ──────────────────────────────────────────────
# 推理辅助函数
# ──────────────────────────────────────────────
def prepare_batch(messages_list, processor, box_op="crop"):
    """
    将一组消息列表转化为模型可接受的 batch dict。

    messages_list: list[list[dict]]，每个元素是一条完整的对话消息
    box_op: 控制如何处理 box，可选 "crop" / "draw" / "none" / "concat"
    """
    image_inputs, id_dict = process_vision_info_with_focal(messages_list, box_op=box_op)
    video_inputs = None

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]

    inputs, crop_or_concat_img_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        id_dict=id_dict,
        replace_two_imgs=set(id_dict.multi_img_texts),
    )

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    attention_mask = inputs.get("attention_mask", None)
    image_grid_thw = inputs.get("image_grid_thw", None)

    batch = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,  # 由 crop/concat 分组传入
        image_grid_thw=image_grid_thw,
        labels=labels,
        has_hard_negative=False,
        id_dict=id_dict,
    )
    # 合入 crop/concat 分组的 pixel_values 与 image_grid_thw
    if crop_or_concat_img_inputs is not None:
        batch.update(crop_or_concat_img_inputs)

    return batch


def tensors_to_device(data, device, dtype=model.dtype):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            if key == "pixel_values":
                data[key] = data[key].to(device).to(dtype)
            else:
                data[key] = data[key].to(device)
    return data


def encode(messages_list, box_op="crop"):
    """对一组消息进行编码，返回 L2 归一化后的 embedding。"""
    batch = prepare_batch(messages_list, processor, box_op=box_op)
    batch = tensors_to_device(batch, DEVICE)

    with torch.no_grad():
        embed = model(**batch, inference=True)  # (batch_size, embed_dim)
    embed = F.normalize(embed, dim=-1)
    return embed


# ──────────────────────────────────────────────
# 构造示例消息：1 个文本查询 vs 2 张候选图片
# ──────────────────────────────────────────────

# 文本查询
text_message = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "A silver kettle on a stand with an arched handle. Find me an everyday image that matches the given caption.\nSummarize above sentence in one word: "},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>."}],
    },
]

# 候选图片 1（无 box）
image_message1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./demo.jpeg", "box": [ 0.295,0.326, 0.372,0.479], "box_op": "crop"}, # POT
            {"type": "text", "text": "\nSummarize above image in one word: "},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>."}],
    },
]

# 候选图片 2（带 box，使用 crop 模式裁剪 focal 区域）
# box 格式: [x1, y1, x2, y2]，坐标为 [0,1] 相对坐标
image_message2_with_box = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./demo.jpeg", "box": [0.165, 0.326, 0.423,0.910], "box_op": "crop"}, # OVEN
            # {"type": "image", "image": "./demo.jpeg", "box": [0.165,0.542, 0.363,0.910], "box_op": "crop"}, # OVEN
            {"type": "text", "text": "\nSummarize above image in one word: "},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>."}],
    },
]


# ──────────────────────────────────────────────
# 执行推理
# ──────────────────────────────────────────────
# 文本 embedding
text_embeds = encode([text_message], box_op="none")
print("text_embeds shape:", text_embeds.shape)

# 图片 embedding（无 box）
image1_embeds = encode([image_message1], box_op="none")
print("image1_embeds shape:", image1_embeds.shape)

# 图片 embedding（带 box，crop 模式）
image2_embeds = encode([image_message2_with_box], box_op="crop")
print("image2_embeds (with box) shape:", image2_embeds.shape)

# 拼接两张图片的 embedding，计算文本与每张图片的相似度
image_embeds = torch.cat([image1_embeds, image2_embeds], dim=0)  # (2, dim)
similarities = (text_embeds @ image_embeds.t()).squeeze(0)        # (2,)

print("\n=== Similarity (text vs 2 images) ===")
print(f"text <-> image1 (no box):    {similarities[0].item():.4f}")
print(f"text <-> image2 (with box):  {similarities[1].item():.4f}")
