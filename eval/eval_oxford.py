
import os
import pickle
import datasets

_VERSION = datasets.Version("1.0.0")

_URLS = {
    "roxford5k": {
        "images": [
            "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
        ],
        "ground_truth": [
            "http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"
        ],
    },
    "rparis6k": {
        "images": [
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz",
            "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz",
        ],
        "ground_truth": [
            "http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl"
        ],
    },
    "revisitop1m": {
        "images": [
            f"http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.{i+1}.tar.gz"
            for i in range(100)
        ]
    },
}

_DESCRIPTION = (
    "Oxford5k, Paris6k, and RevisitOP1M benchmark datasets for image retrieval."
)

_CITATION = """\
@inproceedings{Radenovic2018RevisitingOP,
  title={Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
  author={Filip Radenovic and Ahmet Iscen and Giorgos Tolias and Yannis Avrithis and Ondrej Chum},
  year={2018}
}
"""

BUILDER_CONFIGS = [
    datasets.BuilderConfig(
        name="roxford5k",
        version=_VERSION,
        description="Oxford 5k image retrieval dataset.",
    ),
    datasets.BuilderConfig(
        name="rparis6k",
        version=_VERSION,
        description="Paris 6k image retrieval dataset.",
    ),
    datasets.BuilderConfig(
        name="revisitop1m",
        version=_VERSION,
        description="RevisitOP 1M distractor images.",
    ),
    datasets.BuilderConfig(
        name="oxfordparis",
        version=_VERSION,
        description="Oxford + Paris combined dataset.",
    ),
]
class RevisitOP(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "roxford5k"
    
    # 定义本地基础路径
    _LOCAL_DATA_DIR = "/mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/Oxford5k"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "filename": datasets.Value("string"),
                    "dataset": datasets.Value("string"),
                    "query_id": datasets.Value("int32"),
                    "bbx": datasets.Sequence(datasets.Value("float32")),
                    "easy": datasets.Sequence(datasets.Value("int32")),
                    "hard": datasets.Sequence(datasets.Value("int32")),
                    "junk": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=None,
            homepage="http://cmp.felk.cvut.cz/revisitop/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # 1. 指定本地图片目录和标注文件路径
        # 根据你的描述，所有 jpg 和 pkl 都在该目录下
        image_dir = self._LOCAL_DATA_DIR
        
        # 假设 Oxford5k 的标注文件名是 gnd_roxford5k.pkl
        # 如果文件名不同，请根据实际情况修改
        gt_filename = "gnd_roxford5k.pkl" 
        gt_path = os.path.join(self._LOCAL_DATA_DIR, gt_filename)

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"标注文件未找到: {gt_path}。请确认文件名是否正确。")

        # 2. 返回 SplitGenerator，直接使用本地路径作为 gen_kwargs
        return [
            datasets.SplitGenerator(
                name="qimlist",
                gen_kwargs={
                    "image_dirs": [image_dir],
                    "ground_truth_files": [gt_path],
                    "split_type": "qimlist",
                    "dataset_name": self.config.name,
                },
            ),
            datasets.SplitGenerator(
                name="imlist",
                gen_kwargs={
                    "image_dirs": [image_dir],
                    "ground_truth_files": [gt_path],
                    "split_type": "imlist",
                    "dataset_name": self.config.name,
                },
            ),
        ]

    def _generate_examples(self, image_dirs, ground_truth_files, split_type, dataset_name):
        # 该部分逻辑基本保持不变，因为它已经支持从传入的路径中通过 os.walk 搜索文件 [citation:1][citation:5]
        image_path_mapping = {}
        for image_dir in image_dirs:
            # 遍历本地目录下的所有图片文件 [citation:1][citation:2]
            for root, _, files in os.walk(image_dir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        fpath = os.path.join(root, fname)
                        fname_no_ext = os.path.splitext(fname)[0]
                        image_path_mapping[fname_no_ext] = fpath

        # 加载本地标注文件
        ground_truth_data = []
        for gt_file in ground_truth_files:
            with open(gt_file, "rb") as f:
                gt_data = pickle.load(f)
                ground_truth_data.append(gt_data)

        key = 0
        for gt_data in ground_truth_data:
            imlist = gt_data["imlist"]
            qimlist = gt_data["qimlist"]
            gnd = gt_data["gnd"]

            if split_type == "qimlist":
                for i, query_name in enumerate(qimlist):
                    query_name_no_ext = os.path.splitext(query_name)[0]
                    if query_name_no_ext in image_path_mapping:
                        query_gnd = gnd[i]
                        yield key, {
                            "image": image_path_mapping[query_name_no_ext],
                            "filename": query_name,
                            "dataset": dataset_name,
                            "query_id": i,
                            "bbx": query_gnd.get("bbx", []),
                            "easy": query_gnd.get("easy", []),
                            "hard": query_gnd.get("hard", []),
                            "junk": query_gnd.get("junk", []),
                        }
                        key += 1
            elif split_type == "imlist":
                for i, image_name in enumerate(imlist):
                    image_name_no_ext = os.path.splitext(image_name)[0]
                    if image_name_no_ext in image_path_mapping:
                        yield key, {
                            "image": image_path_mapping[image_name_no_ext],
                            "filename": image_name,
                            "dataset": dataset_name,
                            "query_id": -1,
                            "bbx": [],
                            "easy": [],
                            "hard": [],
                            "junk": [],
                        }
                        key += 1


def compute_oxford_map(similarity, query_dataset, num_db):
    """
    Oxford/Paris revisit 协议下计算 mAP。

    严格按照官方 compute_map / compute_ap 逻辑实现：
      - 正例 (ok)  = easy + hard 对应的 imlist 下标
      - junk       = junk 对应的 imlist 下标
      - junk 的处理方式：统计出现在每个正例之前的 junk 数量，
        从该正例的排名中减去，而非直接删除后重排。

    Args:
        similarity  : torch.Tensor [num_queries, num_db]
        query_dataset : HuggingFace Dataset，包含 easy / hard / junk 字段
        num_db      : gallery 数据库大小

    Returns:
        mAP (float)
    """
    import torch
    import numpy as np

    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity, dtype=torch.float32)

    # ranks: shape [num_db, num_queries]，每列是对应 query 的 gallery 排名（0-based index）
    # 与官方格式保持一致
    ranks = torch.argsort(similarity, dim=1, descending=True).T.numpy()  # [num_db, num_queries]

    nq = len(query_dataset)
    aps = []
    nempty = 0

    for i in range(nq):
        query_info = query_dataset[i]
        qgnd  = np.array(list(query_info["easy"]) + list(query_info["hard"]), dtype=np.int64)
        qgndj = np.array(list(query_info["junk"]),  dtype=np.int64)

        # 无正例则跳过（与官方保持一致：nempty++，不计入平均）
        if qgnd.shape[0] == 0:
            nempty += 1
            continue

        # 正例和 junk 在排名列表中的位置（0-based）
        pos  = np.where(np.isin(ranks[:, i], qgnd))[0]
        junk = np.where(np.isin(ranks[:, i], qgndj))[0]

        # 对每个正例位置，减去其前面出现的 junk 数量（官方算法）
        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] -= k
                ip += 1

        # compute_ap：pos 已是补偿后的 0-based 位置
        ap = 0.0
        for j, p in enumerate(pos):          # j=命中次数-1，p=补偿后位置
            ap += (j + 1) / (p + 1)          # precision@rank = (j+1)/(p+1)
        ap /= len(qgnd)
        aps.append(ap)

    mAP = float(np.mean(aps)) if aps else 0.0
    return mAP


if __name__ == "__main__":
    import argparse
    import sys
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    from accelerate import Accelerator
    from datasets import load_dataset

    # 将项目根目录加入 sys.path，以便导入自定义模块
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_file_path, "../")
    sys.path.append(root_path)

    from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
    from loaders.processor import LemuirProcessor
    from collators.qwen2_vision_process import process_vision_info_with_focal

    BOX_OP = os.environ.get("BOX_OP", "crop")

    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(description="Oxford5k / Paris6k Retrieval Evaluation")
    parser.add_argument("--model_id",          type=str, required=True,  help="微调模型路径")
    parser.add_argument("--original_model_id", type=str, default=None,   help="原始模型路径（用于加载 processor）")
    parser.add_argument("--dataset_name",      type=str, default="roxford5k",
                        choices=["roxford5k", "rparis6k"], help="数据集名称")
    parser.add_argument("--batch_size",        type=int, default=8)
    args = parser.parse_args()

    accelerator = Accelerator()

    # ---------- 加载模型 ----------
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    processor_id = args.original_model_id
    processor  = LemuirProcessor.from_pretrained(processor_id)
    tokenizer  = processor.tokenizer
    tokenizer.padding_side = "left"

    # 注册 <emb> token
    num_new = tokenizer.add_tokens(["<emb>"])
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.config.emb_token_ids = tokenizer.convert_tokens_to_ids(["<emb>"])

    # ---------- 加载数据集 ----------
    dataset_script = "dataset/dataset_oxford5k.py" #os.path.join(current_file_path, "dataset_oxford.py")
    query_hf = load_dataset(dataset_script, name=args.dataset_name, split="qimlist", trust_remote_code=True)
    db_hf    = load_dataset(dataset_script, name=args.dataset_name, split="imlist",  trust_remote_code=True)

    # ---------- PyTorch Dataset ----------
    class OxfordDataset(Dataset):
        def __init__(self, hf_split, use_bbox=False):
            self.data = hf_split
            self.use_bbox = use_bbox  # True → query（使用 bbx 裁剪）

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item   = self.data[idx]
            image  = item["image"].convert("RGB")
            img_w, img_h = image.size
            if self.use_bbox and item["bbx"]:
                x1, y1, x2, y2 = item["bbx"]
                box = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
            else:
                box = [0.0, 0.0, 1.0, 1.0]
            return {"image": image, "box": box}

    query_dataset_pt = OxfordDataset(query_hf, use_bbox=True)
    db_dataset_pt    = OxfordDataset(db_hf,    use_bbox=False)

    # ---------- Collate 函数 ----------
    def collate_fn(batch, mode="db"):
        """
        mode: "query" | "db"
        query 使用 BOX_OP（默认 crop）对 bbox 区域做裁剪；
        db    固定使用 "none"，即输入完整图像。
        """
        messages_list = []
        text_prompt = (
            "Find the most similar image.\nSummarize the image and scentence in one word: "
            # "Retrieve the similar building as the target region. \nSummarize the image and scentence in one word: "
            # "Retrieve the similar building. \nSummarize the image and scentence in one word: "
            if mode == "query"
            else "Summarize the image in one word: "
        )
        cur_box_op = BOX_OP if mode == "query" else "none"
        for item in batch:
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"], "box": item["box"], "box_op": cur_box_op},
                        {"type": "text",  "text": text_prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
            messages_list.append(msgs)

        image_inputs, id_dict = process_vision_info_with_focal(messages_list, box_op=cur_box_op)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        replace_two_imgs = set(id_dict.multi_img_texts) if hasattr(id_dict, "multi_img_texts") else set()
        inputs, crop_or_concat_img_inputs = processor(
            text=texts, images=image_inputs, videos=None,
            padding=True, return_tensors="pt", id_dict=id_dict,
            replace_two_imgs=replace_two_imgs,
        )
        input_ids = inputs["input_ids"]
        labels    = input_ids.clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100

        model_inputs = {
            "input_ids":        inputs["input_ids"],
            "attention_mask":   inputs.get("attention_mask"),
            "pixel_values":     inputs['pixel_values'],
            "labels":           labels,
            "image_grid_thw":   inputs.get("image_grid_thw"),
        }
        model_inputs.update(crop_or_concat_img_inputs)
        return model_inputs, id_dict

    query_loader = DataLoader(
        query_dataset_pt, batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, mode="query"),
    )
    db_loader = DataLoader(
        db_dataset_pt, batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, mode="db"),
    )

    model, query_loader, db_loader = accelerator.prepare(model, query_loader, db_loader)

    # ---------- 提取特征 ----------
    def get_embeddings(dataloader):
        model.eval()
        all_embeds = []
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            model_inputs, id_dict = batch
            model_inputs = {
                k: v.to(accelerator.device)
                for k, v in model_inputs.items()
                if v is not None and isinstance(v, torch.Tensor)
            }
            with torch.no_grad():
                embeds = model(**model_inputs, id_dict=id_dict, inference=True)
                embeds = F.normalize(embeds, dim=-1)
                gathered = accelerator.gather_for_metrics(embeds)
                all_embeds.append(gathered.cpu())
        return torch.cat(all_embeds, dim=0)

    accelerator.print("Extracting query embeddings...")
    query_embeds = get_embeddings(query_loader)

    accelerator.print("Extracting gallery embeddings...")
    db_embeds = get_embeddings(db_loader)

    # ---------- 计算 mAP ----------
    if accelerator.is_main_process:
        query_embeds = query_embeds.to(accelerator.device)
        db_embeds    = db_embeds.to(accelerator.device)

        accelerator.print("Computing similarity matrix...")
        scores = torch.matmul(query_embeds, db_embeds.t())  # [num_queries, num_db]

        accelerator.print("Computing mAP (Oxford revisit protocol)...")
        mAP = compute_oxford_map(scores.cpu(), query_hf, num_db=len(db_hf))

        print(f"\n===== Results on {args.dataset_name} =====")
        print(f"Queries : {len(query_hf)}")
        print(f"Gallery : {len(db_hf)}")
        print(f"mAP     : {mAP:.4f}")

# BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_oxford.py \
#   --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
#   --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs" \
#   --dataset_name roxford5k \
#   --batch_size 50


# BOX_OP='none' LAYERWISE=0 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_oxford.py \
#     --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
#     --model_id "checkpoints/LamRA-Ret" \
#   --dataset_name roxford5k \
#   --batch_size 50


# BOX_OP='none' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_oxford.py   --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/"   --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"   --dataset_name roxford5k   --batch_size 50
# BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_oxford.py   --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/"   --model_id "checkpoints/Lemur_8B_zeroshot-enc"   --dataset_name roxford5k   --batch_size 50