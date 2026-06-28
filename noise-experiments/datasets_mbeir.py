import os
import json
import numpy as np
from torch.utils.data import Dataset
import random


DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

# ──────────────────────────────────────────────
# Box noise 配置（从环境变量读取）
# ──────────────────────────────────────────────
BOX_NOISE_TYPE = os.getenv("BOX_NOISE_TYPE", "none")
# 可选值:
#   "none"           - 不添加噪声
#   "translate"      - 随机方向平移
#   "translate_up"   - 向上平移
#   "translate_down" - 向下平移
#   "translate_left" - 向左平移
#   "translate_right"- 向右平移
#   "scale"          - 随机缩放（放大或缩小）
#   "scale_up"       - 放大
#   "scale_down"     - 缩小
#   "all"            - 同时施加平移 + 缩放

BOX_NOISE_INTENSITY = float(os.getenv("BOX_NOISE_INTENSITY", "0.0"))
# 高斯分布的均值，即噪声幅度的期望值（相对坐标单位）

BOX_NOISE_STD_RATIO = float(os.getenv("BOX_NOISE_STD_RATIO", "0.5"))
# std = intensity × ratio，控制噪声的离散程度


# ──────────────────────────────────────────────
# 噪声核心函数
# ──────────────────────────────────────────────

def _sample_positive_noise(intensity: float, std_ratio: float = BOX_NOISE_STD_RATIO) -> float:
    """
    从 N(intensity, (intensity * std_ratio)^2) 采样正值噪声幅度。
    均值 = intensity，即强度参数直接对应高斯均值。
    负数截断为 0（避免方向反转）。
    """
    std = intensity * std_ratio
    sample = np.random.normal(loc=intensity, scale=std)
    return float(max(0.0, sample))

def add_box_noise(
    box,
    noise_type: str = BOX_NOISE_TYPE,
    intensity: float = BOX_NOISE_INTENSITY,
) -> list:
    """
    对 [0,1] 相对坐标的 bounding box 添加噪声。

    平移语义：intensity 表示平移量占 box 对应方向边长的比例。
              例如 intensity=0.5 → 平移半个 box 宽/高。
    缩放语义：intensity 表示边长缩放变化量的比例（相对原始边长）。
              例如 intensity=0.5 → 边长变为原来的 1.5x 或 0.5x。
    """
    if box is None or noise_type == "none" or intensity <= 0.0:
        return box

    x1, y1, x2, y2 = box

    # 转为中心+尺寸格式
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1

    # ── 平移噪声（相对 box 自身边长） ──────────────────────
    _TRANSLATE_TYPES = {
        "translate", "translate_up", "translate_down",
        "translate_left", "translate_right", "all"
    }
    if noise_type in _TRANSLATE_TYPES:
        # mag ~ N(intensity, (intensity × std_ratio)²), clip ≥ 0
        # mag=1.0 表示平移一个完整的 box 边长
        mag = _sample_positive_noise(intensity)

        if noise_type == "translate_right":
            # 向右平移：偏移量 = mag × box宽度
            cx += mag * w

        elif noise_type == "translate_left":
            # 向左平移：偏移量 = mag × box宽度
            cx -= mag * w

        elif noise_type == "translate_down":
            # 向下平移：偏移量 = mag × box高度
            cy += mag * h

        elif noise_type == "translate_up":
            # 向上平移：偏移量 = mag × box高度
            cy -= mag * h

        else:
            # "translate" / "all"：随机方向角度 θ
            # x方向偏移相对 box宽度，y方向偏移相对 box高度
            # 保证各向同性（小box和大box感受到等比例的扰动）
            angle = np.random.uniform(0, 2 * np.pi)
            cx += mag * np.cos(angle) * w
            cy += mag * np.sin(angle) * h

    # ── 缩放噪声（相对 box 自身边长） ──────────────────────
    _SCALE_TYPES = {"scale", "scale_up", "scale_down", "all"}
    if noise_type in _SCALE_TYPES:
        # mag ~ N(intensity, ...)
        # scale_factor = 1 ± mag，即边长变化 mag 倍
        mag = _sample_positive_noise(intensity)

        if noise_type == "scale_up":
            scale_factor = 1.0 + mag          # 放大
        elif noise_type == "scale_down":
            scale_factor = max(1e-3, 1.0 - mag)   # 缩小
        else:
            # "scale" / "all"：随机放大或缩小
            direction = random.choice([1, -1])
            scale_factor = max(1e-3, 1.0 + direction * mag)

        w *= scale_factor
        h *= scale_factor

    # 转回角点格式
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    # Clip 到 [0, 1]
    x1 = float(np.clip(x1, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))

    # 防止 clip 后 box 退化为点或线
    _EPS = 0.005
    if x2 <= x1:
        mid = (x1 + x2) / 2.0
        x1, x2 = max(0.0, mid - _EPS), min(1.0, mid + _EPS)
    if y2 <= y1:
        mid = (y1 + y2) / 2.0
        y1, y2 = max(0.0, mid - _EPS), min(1.0, mid + _EPS)

    return [x1, y1, x2, y2]


# ──────────────────────────────────────────────
# 工具函数（noise 已集成到 _prepare_data_dict）
# ──────────────────────────────────────────────

def _prepare_data_dict(txt, img_path, image_path_prefix, box=None):
    """构造单条数据字典，并对 box 施加噪声。"""
    img = _load_and_preprocess_image(img_path, image_path_prefix)
    noisy_box = add_box_noise(box)          # ← 统一在此注入噪声
    if img is None:
        return {"txt": txt, "box": noisy_box}
    elif txt == "":
        return {"image": img, "box": noisy_box}
    return {"txt": txt, "image": img, "box": noisy_box}


# ──────────────────────────────────────────────
# Dataset 类
# ──────────────────────────────────────────────

class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        query_data_path: str,
        cand_pool_path: str,
        instructions_path: str,
        image_path_prefix: str,
        tokenizer=None,
        max_length=None,
    ) -> None:
        super().__init__()
        self.query_data        = _load_query_data(query_data_path)
        self.cand_pool         = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions= _load_query_instructions(instructions_path)
        self.tokenizer         = tokenizer
        self.image_path_prefix = image_path_prefix
        self.max_length        = max_length

    def __len__(self) -> int:
        return self.max_length if self.max_length is not None else len(self.query_data)

    def construct_messages(self, data_dict):
        if "txt" in data_dict and "image" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"]},
                    {"type": "text",  "text":  f"{data_dict['txt']}\nSummarize above image and sentence in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        elif "txt" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        else:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"]},
                    {"type": "text",  "text":  "\nSummarize above image in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]

    def _prepare_data_dict(self, txt, img_path, image_path_prefix, box=None):
        """实例方法版，同样在此注入噪声。"""
        img = _load_and_preprocess_image(img_path, image_path_prefix)
        noisy_box = add_box_noise(box)      # ← 噪声注入
        if img is None:
            return {"txt": txt, "box": noisy_box}
        elif txt == "":
            return {"image": img, "box": noisy_box}
        return {"txt": txt, "image": img, "box": noisy_box}

    def get_instance(self, index):
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        query_img_path = mbeir_entry.get('query_img_path', None)
        query_modality = mbeir_entry.get("query_modality", None)
        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None 
        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand              = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_modality     = pos_cand.get("modality", None)
        pos_cand_txt          = format_string(pos_cand.get("txt") or "")

        query_prompt         = _get_random_query_prompt(
            query_dataset_id, query_modality, pos_cand_modality, self.query_instructions
        )
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")

        # 截断防止内存溢出
        query_txt_with_prompt = self.tokenizer.decode(
            self.tokenizer(query_txt_with_prompt, truncation=True, max_length=480,
                           padding=False, return_tensors=None, add_special_tokens=False)["input_ids"]
        )
        pos_cand_txt = self.tokenizer.decode(
            self.tokenizer(pos_cand_txt, truncation=True, max_length=480,
                           padding=False, return_tensors=None, add_special_tokens=False)["input_ids"]
        )

        # box 噪声通过 _prepare_data_dict 自动注入
        query   = self._prepare_data_dict(
            query_txt_with_prompt, query_img_path,
            self.image_path_prefix, mbeir_entry.get("box", None)
        )
        pos_can = self._prepare_data_dict(
            pos_cand_txt, pos_cand.get("img_path", None),
            self.image_path_prefix, pos_cand.get("box", None)
        )
        return {"query": query, "pos_cand": pos_can}

    def __getitem__(self, i):
        instance      = self.get_instance(i)
        query_message = self.construct_messages(instance["query"])
        cand_message  = self.construct_messages(instance["pos_cand"])
        return query_message, cand_message


BOXOP = os.getenv("BOXOP", "draw")


class QueryDataset(Dataset):
    def __init__(self, query_data_path, cand_pool_path, instructions_path, image_path_prefix):
        super().__init__()
        self.query_data         = _load_query_data(query_data_path)
        self.cand_pool          = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix  = image_path_prefix

    def __len__(self):
        return len(self.query_data)

    def construct_messages(self, data_dict):
        if "txt" in data_dict and "image" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"],
                     "box": data_dict["box"], "box_op": BOXOP},
                    {"type": "text",  "text":  f"{data_dict['txt']}\nSummarize above image and sentence in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        elif "txt" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        else:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"],
                     "box": data_dict["box"], "box_op": BOXOP},
                    {"type": "text",  "text":  "\nSummarize above image in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]

    def get_instance(self, index):
        mbeir_entry      = self.query_data[index]
        query_txt        = mbeir_entry.get("query_txt") or ""
        query_img_path   = mbeir_entry.get("query_img_path", None)
        query_modality   = mbeir_entry.get("query_modality", None)
        qid              = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None

        pos_cand_list         = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand              = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_modality     = pos_cand.get("modality", None)

        query_prompt          = _get_random_query_prompt(
            query_dataset_id, query_modality, pos_cand_modality, self.query_instructions
        )
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")

        # box 噪声通过 _prepare_data_dict 自动注入
        instance = _prepare_data_dict(
            query_txt_with_prompt, query_img_path,
            self.image_path_prefix, mbeir_entry.get("box", None)
        )
        instance["qid"] = hash_qid(qid)
        return instance

    def __getitem__(self, i):
        query         = self.get_instance(i)
        qid           = query["qid"]
        query_message = self.construct_messages(query)
        return query_message, qid


class CandidateDataset(Dataset):
    def __init__(self, query_data_path, cand_pool_path, instructions_path, image_path_prefix):
        super().__init__()
        self.query_data         = _load_query_data(query_data_path)
        self.cand_pool          = _load_cand_pool(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix  = image_path_prefix

    def __len__(self):
        return len(self.cand_pool)

    def construct_messages(self, data_dict):
        if "txt" in data_dict and "image" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"],
                     "box": data_dict["box"], "box_op": BOXOP},
                    {"type": "text",  "text":  f"{data_dict['txt']}\nSummarize above image and sentence in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        elif "txt" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        else:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"],
                     "box": data_dict["box"], "box_op": BOXOP},
                    {"type": "text",  "text":  "\nSummarize above image in one word: "},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]

    def get_instance(self, index):
        entry       = self.cand_pool[index]
        img_path    = entry.get("img_path", None)
        img         = _load_and_preprocess_image(img_path, self.image_path_prefix)
        did         = entry.get("did", None)
        cand_txt    = format_string(entry.get("txt") or "")
        cand_mod    = entry.get("modality", None)

        # box 噪声通过 add_box_noise 注入
        noisy_box = add_box_noise(entry.get("box", None))

        if img is not None and cand_txt:
            instance = {"txt": cand_txt, "image": img, "modality": cand_mod}
        elif img is not None:
            instance = {"image": img, "modality": cand_mod}
        else:
            instance = {"txt": cand_txt, "modality": cand_mod}

        instance["did"] = hash_did(did)
        instance["box"] = noisy_box
        return instance

    def __getitem__(self, i):
        candidate        = self.get_instance(i)
        did              = candidate["did"]
        candidate_message= self.construct_messages(candidate)
        return candidate_message, did


class MbeirLanguageDataset(LazySupervisedDataset):
    def __init__(self, query_data_path, cand_pool_path, instructions_path,
                 image_path_prefix, tokenizer=None, max_length=None):
        super().__init__(query_data_path, cand_pool_path, instructions_path,
                         image_path_prefix, tokenizer, max_length)

    def construct_messages(self, data_dict, pos_cand_dict):
        if "image" in data_dict and "txt" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"], "box": data_dict["box"]},
                    {"type": "text",  "text":  f"{data_dict['txt']}\nDescribe the whole image through a caption."},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": pos_cand_dict["txt"]}]},
            ]
        elif "image" in data_dict:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": data_dict["image"], "box": data_dict["box"]},
                    {"type": "text",  "text":  "Describe the whole image through a caption."},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": pos_cand_dict["txt"]}]},
            ]
        else:
            raise ValueError("Data dict must contain 'image' for MbeirLanguageDataset.")

    def get_instance(self, index):
        mbeir_entry      = self.query_data[index]
        query_txt        = mbeir_entry.get("query_txt") or ""
        query_img_path   = mbeir_entry.get("query_img_path", None)
        qid              = mbeir_entry.get("qid", None)

        pos_cand_list         = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand              = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_txt          = format_string(pos_cand.get("txt") or "")

        query_txt_without_prompt = format_string(query_txt)

        query_txt_without_prompt = self.tokenizer.decode(
            self.tokenizer(query_txt_without_prompt, truncation=True, max_length=480,
                           padding=False, return_tensors=None, add_special_tokens=False)["input_ids"]
        )
        pos_cand_txt = self.tokenizer.decode(
            self.tokenizer(pos_cand_txt, truncation=True, max_length=480,
                           padding=False, return_tensors=None, add_special_tokens=False)["input_ids"]
        )

        # box 噪声通过 _prepare_data_dict 自动注入
        query   = _prepare_data_dict(
            query_txt_without_prompt, query_img_path,
            self.image_path_prefix, mbeir_entry.get("box", None)
        )
        pos_can = _prepare_data_dict(
            pos_cand_txt, pos_cand.get("img_path", None),
            self.image_path_prefix, pos_cand.get("box", None)
        )
        return {"query": query, "pos_cand": pos_can}

    def __getitem__(self, i):
        instance  = self.get_instance(i)
        message1  = self.construct_messages(instance["query"], instance["pos_cand"])
        j         = i + self.max_length
        instance  = self.get_instance(j)
        message2  = self.construct_messages(instance["query"], instance["pos_cand"])
        return message1, message2


# ──────────────────────────────────────────────
# 其余工具函数（与原始代码相同）
# ──────────────────────────────────────────────

def _load_data(data_path):
    assert data_path.endswith(".jsonl"), f"{data_path} is not a jsonl file"
    return _load_data_jsonl(data_path)

def _load_query_data(query_data_path):
    return _load_data(query_data_path)

def _load_cand_pool_as_dict(cand_pool_data_path):
    cand_pool = _load_data(cand_pool_data_path)
    return {e["did"]: e for e in cand_pool}

def _load_cand_pool(cand_pool_data_path):
    return _load_data(cand_pool_data_path)

def _load_query_instructions(instructions_path):
    assert os.path.exists(instructions_path)
    assert instructions_path.endswith(".tsv")
    prompts_dict = {}
    with open(instructions_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts_dict[key] = [p for p in parts[4:] if p]
    return prompts_dict

def _get_random_cand(cand_list):
    return random.choice(cand_list)

def format_string(s):
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:
        s = s[0].upper() + s[1:]
        s = s + "." if s[-1] not in [".", "?", "!"] else s
    return s

def _get_random_query_prompt(dataset_id, query_modality, cand_modality, query_instructions):
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    prompts = query_instructions.get(key, [])
    assert prompts, f"Cannot find prompts for {key}"
    return format_string(random.choice(prompts))

def _load_and_preprocess_image(query_img_path, image_path_prefix):
    if not query_img_path:
        return None
    return os.path.join(image_path_prefix, query_img_path)

def _load_data_jsonl(datapath):
    with open(datapath) as f:
        return [json.loads(line) for line in f]

def hash_qid(qid):
    dataset_id, within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + within_id

def unhash_qid(hashed_qid):
    return f"{hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND}:{hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND}"

def hash_did(did):
    dataset_id, within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + within_id

def unhash_did(hashed_did):
    return f"{hashed_did // DATASET_CAN_NUM_UPPER_BOUND}:{hashed_did % DATASET_CAN_NUM_UPPER_BOUND}"
