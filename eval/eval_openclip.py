from open_clip import create_model_and_transforms, get_tokenizer
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import defaultdict
from accelerate import Accelerator
import accelerate
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from dataset.datasets_mbeir import QueryDataset, CandidateDataset


# Data collation constants
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import ast
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


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


def compute_recall_at_k_rewrite(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents

    relevant_docs_set = set(relevant_docs)
    retrieved_indices_set = set(retrieved_indices)
    result = []
    for target_doc in relevant_docs_set:
        other_relevant_docs = relevant_docs_set - {target_doc}
        num_other_relevant_retrieved = len(retrieved_indices_set.intersection(other_relevant_docs))
        dynamic_k = k + num_other_relevant_retrieved
        top_dynamic_k_retrieved_set = set(retrieved_indices[:dynamic_k])
        
        if target_doc in top_dynamic_k_retrieved_set:
            result.append(1.0)
        else:
            result.append(0.0)
            
    return result

class OpenCLIPQueryDataCollator:
    def __init__(self, processor, image_path_prefix, query_modal="image,text"):
        self.processor = processor
        self.image_path_prefix = image_path_prefix
        self.query_modal = query_modal.split(',') if query_modal else ["image", "text"]
        self.use_image = "image" in self.query_modal
        self.use_text = "text" in self.query_modal

    def crop_image_with_box(self, image, box):
        """根据box信息对图片进行crop"""
        if box is None:
            return image
        
        width, height = image.size
        x0 = width * box[0]
        y0 = height * box[1]
        x1 = width * box[2]
        y1 = height * box[3]
        
        # 确保坐标在有效范围内
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        
        # 如果box有效，进行crop
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        else:
            return image

    def process_image_with_box_op(self, image, box, box_op):
        """根据box_op类型处理图片"""
        if box_op == "crop":
            return self.crop_image_with_box(image, box)
        elif box_op == "none" or box_op is None:
            return image
        else:
            # 默认情况，如果不认识的box_op，使用crop
            return self.crop_image_with_box(image, box)

    def __call__(self, batch):
        images = []
        texts = []
        query_ids = []
        
        for item in batch:
            # item is a tuple: (query_message, qid)
            query_message, qid = item
            query_ids.append(qid)
            
            # 提取文本、图片、box和box_op信息
            text_content = ""
            image_path = None
            box = None
            box_op = None
            
            # 从消息中提取内容
            if query_message and len(query_message) > 0:
                user_content = query_message[0].get('content', [])
                for content_item in user_content:
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image_path = content_item.get('image', None)
                        box = content_item.get('box', None)
                        box_op = content_item.get('box_op', None)
            
            # 清理文本内容
            text_content = text_content.replace("\nSummarize above sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image and sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image in one word: ", "")
            
            # 根据modal设置决定是否使用文本
            if self.use_text:
                texts.append(text_content)
            else:
                texts.append("")  # 空文本
            
            # 根据modal设置决定是否使用图片  
            if self.use_image and image_path:
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.image_path_prefix, image_path)
                image = Image.open(image_path).convert('RGB')
                # 根据box_op进行相应处理
                image = self.process_image_with_box_op(image, box, box_op)
                images.append(image)
            else:
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        # 使用OpenCLIP processor处理图片和文本
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        inputs.update({'query_ids': query_ids})

        return inputs


class OpenCLIPCandidateDataCollator:
    def __init__(self, processor, image_path_prefix, cand_modal="image,text"):
        self.processor = processor
        self.image_path_prefix = image_path_prefix
        self.cand_modal = cand_modal.split(',') if cand_modal else ["image", "text"]
        self.use_image = "image" in self.cand_modal
        self.use_text = "text" in self.cand_modal

    def crop_image_with_box(self, image, box):
        """根据box信息对图片进行crop"""
        if box is None:
            return image
        
        width, height = image.size
        x0 = width * box[0]
        y0 = height * box[1]
        x1 = width * box[2]  
        y1 = height * box[3]
        
        # 确保坐标在有效范围内
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        
        # 如果box有效，进行crop
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        else:
            return image

    def process_image_with_box_op(self, image, box, box_op):
        """根据box_op类型处理图片"""
        if box_op == "crop":
            return self.crop_image_with_box(image, box)
        elif box_op == "none" or box_op is None:
            return image
        else:
            # 默认情况，如果不认识的box_op，使用crop
            return self.crop_image_with_box(image, box)

    def __call__(self, batch):
        images = []
        texts = []
        candidate_ids = []
        
        for item in batch:
            # item is a tuple: (candidate_message, did)
            candidate_message, did = item
            candidate_ids.append(did)
            
            # 提取文本、图片、box和box_op信息
            text_content = ""
            image_path = None
            box = None
            box_op = None
            
            # 从消息中提取内容
            if candidate_message and len(candidate_message) > 0:
                user_content = candidate_message[0].get('content', [])
                for content_item in user_content:
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image_path = content_item.get('image', None)
                        box = content_item.get('box', None)
                        box_op = content_item.get('box_op', None)

            # 清理文本内容
            text_content = text_content.replace("\nSummarize above sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image and sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image in one word: ", "")
            
            # 根据modal设置决定是否使用文本
            if self.use_text:
                texts.append(text_content)
            else:
                texts.append("")  # 空文本
            
            # 根据modal设置决定是否使用图片
            if self.use_image and image_path:
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.image_path_prefix, image_path)
                image = Image.open(image_path).convert('RGB')
                # 根据box_op进行相应处理
                image = self.process_image_with_box_op(image, box, box_op)
                images.append(image)
            else:
                # 不使用图片或图片路径为空，使用空白图片
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        # 使用OpenCLIP processor处理图片和文本
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        inputs.update({'candidate_ids': torch.tensor([int(x) for x in candidate_ids])})
        
        return inputs


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/home/kecheng/.cache/clip/",
        help="Where to store files cache_dir.",
    )
    parser.add_argument(
        "--root_filename",
        type=str,
        default='./datasets',
        help="Where to store data.",
    )
    parser.add_argument(
        "--json_root",
        type=str,
        default='/input_ssd/datasets/',
        help="Where to store json data.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="torch distributed needed.",
    )
    parser.add_argument(
        "--num_text",
        type=int,
        default=6,
        help="Sampling the number of sub-captions.",
    )
    parser.add_argument(
        "--split_json_size",
        type=int,
        default=1,
        help="Number of sampling in getitem of dataloader.",
    )
    parser.add_argument(
        "--merged_num",
        type=int,
        default=1,
        help="The used number of short captions in long caption.",
    )
    parser.add_argument(
        "--context_length", type=int, default=77, help="Used token number."
    )
    parser.add_argument(
        "--sgl_delta",
        type=float,
        default=0.0,
        help="Threshhold of grouping for subcaption-specific grouping loss.",
    )
    parser.add_argument(
        "--sgl_norm",
        type=str,
        default='minmax',
        help="Norm function for subcaption-specific grouping loss.",
    )
    parser.add_argument(
        "--clip_lossweight",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--sgl_lossweight",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--meta-nouns",
        type=str,
        default=None,
        help="Where to store meta file of nouns.",
    )
    parser.add_argument(
        "--tag-mode",
        type=str,
        default='mixed',
        help="use which feature to calculate multi-tag loss.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "json", "txt"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--use_longcap",
        default=False,
        action="store_true",
        help="Whether to use longcapion file to parse csv data."
    )
    parser.add_argument(
        "--use_synimg",
        default=False,
        action="store_true",
        help="Whether to use syn. images to train model."
    )
    parser.add_argument(
        "--use_multipos_loss",
        default=False,
        action="store_true",
        help="Whether to use multi-pos loss to train model."
    )
    parser.add_argument(
        "--use_finegrained_loss",
        default=False,
        action="store_true",
        help="Whether to use fine-grained loss to train model."
    )
    parser.add_argument(
        "--use_declip_loss",
        default=False,
        action="store_true",
        help="Whether to use declip loss to train model."
    )
    parser.add_argument(
        "--use_ot_loss",
        default=False,
        action="store_true",
        help="Whether to use OT loss to train model."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--margin", type=float, default=0.25, help="Margin for circle loss.")
    parser.add_argument(
        "--gamma", type=int, default=32, help="Gamma for circle loss."
    )
   
    parser.add_argument(
        "--use-dataaug",
        default=False,
        action="store_true",
        help="Whether to use dataaug.")
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--vit-backbone",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--model-name",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )

    parser.add_argument(
        "--dci",
        type=str,
        default=None,
        help="Path to dci data for conducting zero shot evaluation.",
    )

    # Evaluation-specific arguments
    parser.add_argument(
        '--query_data_path', 
        type=str, 
        default=None,
        help='Path to query data file'
    )
    parser.add_argument(
        '--cand_pool_path', 
        type=str, 
        default=None,
        help='Path to candidate pool data file'
    )
    parser.add_argument(
        '--instructions_path', 
        type=str, 
        default=None,
        help='Path to instructions file'
    )
    parser.add_argument(
        '--qrels_path', 
        type=str, 
        default=None,
        help='Path to qrels file'
    )
    parser.add_argument(
        '--query_cand_pool_path', 
        type=str, 
        default=None,
        help='Path to query candidate pool'
    )
    parser.add_argument(
        '--image_path_prefix', 
        type=str, 
        default=None,
        help='Prefix path for images'
    )
    parser.add_argument(
        '--query_modal', 
        type=str, 
        default='image,text', 
        help='Modalities for query processing: "image", "text", or "image,text"'
    )
    parser.add_argument(
        '--cand_modal', 
        type=str, 
        default='image,text', 
        help='Modalities for candidate processing: "image", "text", or "image,text"'
    )

    args = parser.parse_args(args)

    return args


def get_embedding(model, batch, modals):
    """根据指定的模态组合生成embedding"""
    image_features = model.module.encode_image(batch['image']) 
    text_features = model.module.encode_text(batch['text']) 
    
    if "image" in modals and "text" in modals:
        embed = (image_features + text_features) / 2
    elif "image" in modals:
        embed = image_features
    elif "text" in modals:
        embed = text_features
    else:
        embed = (image_features + text_features) / 2
        
    return F.normalize(embed, dim=-1)


def main(args):
    # Create model and transforms
    model_kwargs = {}
    model_kwargs['tag_mode'] = args.tag_mode
    model_kwargs['cache_dir'] = args.cache_dir
    model, _, preprocess_val = create_model_and_transforms(
        # args.vit_backbone,
        # args.model_name,
    #     precision=args.precision,
    #     device="cuda",
    #     jit=args.torchscript,
    #     force_quick_gelu=args.force_quick_gelu,
    #     force_custom_text=args.force_custom_text,
    #     force_patch_dropout=args.force_patch_dropout,
    #     force_image_size=args.force_image_size,
    #     image_mean=args.image_mean,
    #     image_std=args.image_std,
    #     image_interpolation=args.image_interpolation,
    #     image_resize_mode=args.image_resize_mode,  # only effective for inference
    #     pretrained_image=args.pretrained_image,
    #     output_dict=True,
    #     use_meta_noun=args.meta_nouns,
    #     **model_kwargs,
    # )
            "EVA02-CLIP-L-14-336",
            "eva",
            "amp",
            device="cuda",
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
            cache_dir=args.model_name,
            det_image_size=336,
            dataset_type="grid_distill",
    )
    # Get tokenizer
    tokenizer = get_tokenizer(args.vit_backbone)
    
    # Create processor-like object for OpenCLIP (simple wrapper)
    class OpenCLIPProcessor:
        def __init__(self, preprocess_fn, tokenizer):
            self.preprocess_fn = preprocess_fn
            # if isinstance(preprocess_fn, list):
            self.preprocess_fn = preprocess_fn[0]
            self.tokenizer = tokenizer
            
        def __call__(self, text=None, images=None, return_tensors="pt"):
            result = {}
            if images is not None:
                if isinstance(images, list):
                    processed_images = []
                    for img in images:
                        processed_images.append(self.preprocess_fn(img))

                    result['image'] = torch.stack(processed_images, dim=0)
                else:
                    result['image'] = self.preprocess_fn(images).unsqueeze(0)
            
            if text is not None:
                if isinstance(text, list):
                    tokenized = self.tokenizer(text)
                else:
                    tokenized = self.tokenizer([text])
                result['text'] = tokenized
            
            return result
    
    processor = OpenCLIPProcessor(preprocess_val, tokenizer)

    # Parse modal parameters
    query_modals = args.query_modal.split(',') if args.query_modal else ["image", "text"]
    cand_modals = args.cand_modal.split(',') if args.cand_modal else ["image", "text"]

    # Create datasets
    query_dataset = QueryDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.query_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix
    )

    cand_dataset = CandidateDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix
    )

    # Create data collators
    query_data_collator = OpenCLIPQueryDataCollator(
        processor=processor, 
        image_path_prefix=args.image_path_prefix, 
        query_modal=args.query_modal
    )
    cand_data_collator = OpenCLIPCandidateDataCollator(
        processor=processor, 
        image_path_prefix=args.image_path_prefix, 
        cand_modal=args.cand_modal
    )

    # Create data loaders
    query_dataloader = DataLoader(
        query_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        shuffle=False, 
        collate_fn=query_data_collator
    )
    candidate_dataloader = DataLoader(
        cand_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        shuffle=False, 
        collate_fn=cand_data_collator
    )

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process
    model.to(torch.float32)
    model.eval()

    def tensors_to_device(data, device, dtype=model.dtype if hasattr(model, 'dtype') else torch.float16):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'image':
                    data[key] = data[key].to(device).to(torch.float32)
                else:
                    data[key] = data[key].to(device)
        return data 

    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
        
        # Process query data
        for batch in tqdm(query_dataloader, disable=not is_main_process, desc="Processing queries"):
            batch_query_ids = batch.pop('query_ids')
            batch = tensors_to_device(batch, device)
            # Generate embedding based on query_modal settings
            query_embed = get_embedding(model, batch, query_modals)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

        # Process candidate data
        for batch in tqdm(candidate_dataloader, disable=not is_main_process, desc="Processing candidates"):
            batch_candidate_ids = batch.pop('candidate_ids')
            batch = tensors_to_device(batch, device)
            
            # Generate embedding based on cand_modal settings
            candidate_embed = get_embedding(model, batch, cand_modals)
            
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids.tolist())
            candidate_features.append(candidate_embed)

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    if is_main_process:
        # Compute retrieval scores and rankings
        index = []
        scores = []
        for i in range(len(query_features)):
            query_feature = query_features[i:i+1]
            score = query_feature @ candidate_features.T # (1, num_candidate)
            topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
            topk_indexes = topk_indexes.squeeze().tolist()
            index.append(topk_indexes)
            scores.append(topk_score.tolist())

        cand_names = np.array([[unhash_did(candidate_ids[item]) for item in row] for row in index])
        query_names = [unhash_qid(item) for item in query_ids]

        # Create results directory
        save_dir_name = "./OpenCLIP_Ret_eval_results"
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        
        # Load qrels and compute metrics
        qrel, qid_to_taskid = load_qrel(args.qrels_path)

        k_lists = [1, 5, 10, 50]
        res = {}

        for k in k_lists:
            res[f'recall_{k}'] = []

        for ind, query_name in enumerate(tqdm(query_names, desc="Computing metrics")):
            relevant_docs = qrel.get(query_name, [])
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k_rewrite(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].extend(recall_at_k)

        # Print and save results
        for k in k_lists:
            recall_val = sum(res[f'recall_{k}']) / len(res[f'recall_{k}']) if res[f'recall_{k}'] else 0
            print(f"recall_at_{k} = {recall_val}")

        model_name = args.vit_backbone + "_" + (args.model_name.replace('/', '_') if args.model_name else 'pretrained')
        result_file = f"{save_dir_name}/{model_name}_results.txt"
        
        with open(result_file, 'a') as f:
            f.write(args.qrels_path + '\n')
            for k in k_lists:
                recall_val = sum(res[f'recall_{k}']) / len(res[f'recall_{k}']) if res[f'recall_{k}'] else 0
                f.write(f"recall_at_{k} = {recall_val}" + '\n')
        print(f"Write output to {result_file}")


if __name__ == "__main__":
    args = parse_args(None)
    main(args)