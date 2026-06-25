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
MODEL_ID = "code-kunkun/LamRA-Ret"   # 可替换为你本地微调后的模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ──────────────────────────────────────────────
# 加载模型与处理器
# ──────────────────────────────────────────────
model = Qwen2VLRetForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).to(DEVICE)

processor = LemuirProcessor.from_pretrained(MODEL_ID)
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
# 构造示例消息
# ──────────────────────────────────────────────

# 1) 纯图像消息（无 box）
image_message = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./demo.jpeg"},
            {"type": "text", "text": "Find an image caption describing the following everyday image."},
            {"type": "text", "text": "\nSummarize above image and sentence in one word: "},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>."}],
    },
]

# 2) 图像消息 + box（box_op="crop" 会裁剪出 box 区域作为 focal 信息）
# box 格式: [x1, y1, x2, y2]，坐标为 [0,1] 相对坐标
image_message_with_box = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./demo.jpeg",
             "box": [0.1, 0.2, 0.6, 0.8], "box_op": "crop"},
            {"type": "text", "text": "Find an image caption describing the following everyday image."},
            {"type": "text", "text": "\nSummarize above image and sentence in one word: "},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>."}],
    },
]

# 3) 纯文本消息
text_message1 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "a dog and a woman are playing on the bench.\nSummarize above sentence in one word: "},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>."}],
    },
]

text_message2 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "a dog.\nSummarize above sentence in one word: "},
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
# 图像 embedding（无 box）
image_embeds = encode([image_message], box_op="none")
print("image_embeds shape:", image_embeds.shape)

# 图像 embedding（带 box，使用 crop 模式）
image_embeds_box = encode([image_message_with_box], box_op="crop")
print("image_embeds_with_box shape:", image_embeds_box.shape)

# 文本 embedding
text_embeds = encode([text_message1, text_message2], box_op="none")
print("text_embeds shape:", text_embeds.shape)

# 计算相似度
print("\n=== Similarity (image vs texts) ===")
print("image <-> text1:", (image_embeds @ text_embeds.t())[0, 0].item())
print("image <-> text2:", (image_embeds @ text_embeds.t())[0, 1].item())

if image_embeds_box is not None:
    print("\n=== Similarity (image_with_box vs texts) ===")
    print("image_box <-> text1:", (image_embeds_box @ text_embeds.t())[0, 0].item())
    print("image_box <-> text2:", (image_embeds_box @ text_embeds.t())[0, 1].item())
