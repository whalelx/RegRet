import os
import torch
from tqdm import tqdm
from accelerate import Accelerator
import torch.nn.functional as F
import json
import sys

from datasets import load_dataset, concatenate_datasets
from PIL import Image


BOX_OP = os.environ.get("BOX_OP", "crop")

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)

from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from loaders.processor import LemuirProcessor
from collators.qwen2_vision_process import process_vision_info_with_focal


# --- 全局变量与配置 ---
emb_data_func = None
accelerator = Accelerator()

# --- 工具函数 (来自第一部分代码) ---

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [b[key] for b in batch]
    return collated_batch


def log_to_file(data, metrics, checkpoint_name, difficulty=['easy', 'medium', 'hard']):
    # 保持原样
    if data == 'flickr30k' or data == 'coco':
        output = f"{data}: image R@1 {metrics['image_retrieval_recall@1']:.4f} text R@1 {metrics['text_retrieval_recall@1']:.4f} \n"
        output += f"{data}: image R@5 {metrics['image_retrieval_recall@5']:.4f} text R@5 {metrics['text_retrieval_recall@5']:.4f} \n"
        output += f"{data}: image R@10 {metrics['image_retrieval_recall@10']:.4f} text R@10 {metrics['text_retrieval_recall@10']:.4f} \n"
    else:
        output = ''
        if 'sorce' in data:
            for level in difficulty:
                output += f"{data}: {level} R@1 {metrics[f'[{level}] image_retrieval_recall@1']:.4f} \n"
                output += f"{data}: {level} R@5 {metrics[f'[{level}] image_retrieval_recall@5']:.4f} \n"
                output += f"{data}: {level} R@10 {metrics[f'[{level}] image_retrieval_recall@10']:.4f} \n"
        else:
            output += f"{data}: image R@1 {metrics['image_retrieval_recall@1']:.4f} \n"
            output += f"{data}: image R@5 {metrics['image_retrieval_recall@5']:.4f} \n"
            output += f"{data}: image R@10 {metrics['image_retrieval_recall@10']:.4f} \n"
        output += f"{data}: text R@1 {metrics['text_retrieval_recall@1']:.4f} \n"
        output += f"{data}: text R@5 {metrics['text_retrieval_recall@5']:.4f} \n"
        output += f"{data}: text R@10 {metrics['text_retrieval_recall@10']:.4f} \n"

    # if checkpoint_name is not None:
    #     with open(checkpoint_name, 'a') as f:
    #         print(output, file=f)
    print(output)
    return output


# --- 模型初始化 (重构以支持Qwen2VLRet) ---

def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    if len(emb_tokens) == num_new_tokens:
        print(f"Added {num_new_tokens} new tokens: {emb_tokens}")
    
    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids
    model.resize_token_embeddings(len(tokenizer))


def init_model_and_transform(model_id, original_model_id=None):
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    tokenizer_source = original_model_id if original_model_id else model_id
    processor = LemuirProcessor.from_pretrained(tokenizer_source)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    
    add_embed_token(tokenizer, model)
    model.eval()
    return model, processor


# --- Embedding生成 (核心重构部分) ---

def _run_model_batch(model, processor, device, messages_list):
    """
    内部辅助函数：处理Batch Messages并生成Embeddings
    对应原代码中的 run_model，但适配了batch处理逻辑
    """
    # 1. Vision Info Processing
    # 注意：process_vision_info_with_focal 需要处理 batch 的消息列表
    # 这里假设该函数可以处理或我们在循环中处理。为了效率，尽量batch处理。
    # 如果 process_vision_info_with_focal 不支持 batch list，可能需要降级为循环处理。
    # 参考 eval_fgovd 逻辑，它对一个 list 运行。我们假设它返回聚合的 inputs。
    
    # 鉴于原代码调用方式，这里可能需要遍历每个sample的messages构建input，然后collate。
    # 为了简化，这里采用迭代方式，如果batch_size小（如4），开销可控。
    # 若需极致优化，需自定义 collate_fn 将所有 messages 预处理对齐。
    
    all_embeds = []
    
    # 鉴于 eval_fgovd 代码中 process_vision_info_with_focal 接收的是 list of messages，
    # 这里我们假设它一次处理一个样本。
    # 但为了利用 batch，我们需要先构建好 inputs。
    
    batch_inputs = {}
    
    # 预处理所有样本
    preprocessed_items = []
    for messages in messages_list:
        # messages 结构: [[{role: user, content: ...}, {role: assistant, ...}]]
        # process_vision_info_with_focal 期望传入 messages (List[Dict])
        image_inputs, id_dict = process_vision_info_with_focal(messages, box_op=BOX_OP)
        
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        replace_two_imgs = set(getattr(id_dict, 'multi_img_texts', []))

        # Processor 处理单个样本
        inputs, crop_or_concat_img_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            id_dict=id_dict,
            replace_two_imgs=replace_two_imgs
        )

        # 对齐 fgovd：生成 labels（pad 位置设为 -100）
        input_ids = inputs.get('input_ids')
        labels = None
        if input_ids is not None:
            labels = input_ids.clone()
            pad_id = processor.tokenizer.pad_token_id
            labels[labels == pad_id] = -100
            inputs['labels'] = labels

        # 更新到 dict 结构中，同时保存 id_dict
        item_input = {k: v for k, v in inputs.items() if v is not None}
        item_input.update(crop_or_concat_img_inputs)
        item_input['id_dict'] = id_dict  # 保存 id_dict 供后续使用
        preprocessed_items.append(item_input)
    
    # 简单的 Collate: 对于 size 不一的 tensor (如 pixel_values), 通常 processor 已经 pad 好了
    # 但 processor 在 batch=False 时返回的是单样本 tensor。
    # 这里需要手动 pad。为了代码简洁，这里假设 batch_size=1 的方式调用，
    # 或者在 dataloader 层面处理 collate。
    # 鉴于 emb_data 函数内部处理了 dataloader，我们在 emb_data 内部直接调用 batch 处理逻辑。
    
    # 下面这段逻辑是为了适配 _run_model_batch 可能会被传入多个 messages 的情况
    # 但实际在 emb_data 中我们会控制 batch size。
    
    # 重新聚合 inputs (简单的 list of dicts -> dict of batched tensors)
    # 注意：pixel_values 可能是 nested tensor 或 list，Qwen2VL 处理方式特殊
    if len(preprocessed_items) == 0:
        return torch.tensor([]).to(device)

    # 这里为了适配性，使用最简单的 padding 策略：
    # 如果 processor 返回的是 batched tensors (return_tensors='pt'), 我们需要将其 cat 起来
    # 但 processor 每次只处理一个样本。
    # 我们这里采用循环处理的方式，逻辑最清晰且不易出错。
    
    # 重新思考优化：既然 batch 可能很难手动 collate (涉及复杂的 image grid thw),
    # 我们直接循环处理每个样本，然后 cat 结果。
    # 这适用于 batch_size 较小 (如 4, 8) 的情况。
    
    for item_input in preprocessed_items:
        id_dict = item_input.get('id_dict')
        
        # 构造模型输入
        model_inputs = dict(
            input_ids=item_input.get('input_ids'),
            attention_mask=item_input.get('attention_mask'),
            pixel_values=item_input.get('pixel_values'),
            image_grid_thw=item_input.get('image_grid_thw'),
            labels=item_input.get('labels'),
        )
        model_inputs.update(item_input.get('crop_or_concat_img_inputs'))

        model_inputs = {k: v.to(device) for k, v in model_inputs.items() if v is not None and isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            # 传入 id_dict，参考 fgovd.py 的做法
            embed = model(**model_inputs, id_dict=id_dict, inference=True)
            all_embeds.append(embed)
            
    return torch.cat(all_embeds, dim=0)


def emb_data(model, processor, dataset, device,
             emb_type='text', prompt=None, bsz=4,
             text_column='caption', img_column='img'):
    """
    重写的 embedding 生成函数。
    保留了原函数的接口，内部使用 message 构造和 run_model 逻辑。
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bsz,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn # 返回 list of dicts
    )
    # dataloader = accelerator.prepare(dataloader) # 如果使用多卡，需要 prepare
    
    embs = []
    bar = tqdm(total=len(dataloader), desc=f"Embedding {emb_type}")
    
    for batch in dataloader:
        # batch 是 dict，每个 value 是 list (因为 batch_size > 1)
        # 我们需要将 batch 拆解并构建 messages
        
        batch_size_current = len(batch[img_column]) # 以 img 列为准
        
        messages_list = []
        
        # 构造 Prompt
        # prompt 来自外部，例如 "Summarize above image in one word: "
        # 我们需要将其填入 chat template
        
        for i in range(batch_size_current):
            if emb_type == 'text':
                # 文本 Embedding
                text_content = batch[text_column][i]
                # SORCE 数据中 text 可能是 list
                if isinstance(text_content, list):
                    # 如果是 list，通常只取第一个或者生成多个 embedding?
                    # 原代码逻辑：如果是 list，则在 input_texts 展开处理
                    # 这里为了简化，假设我们处理单条，或者在外部把 dataset flatten
                    # 暂时只取第一条 text
                    text_content = text_content[0] 
                
                # 构造 messages
                # 格式: user: <sent>\n prompt -> assistant: <emb>
                # 注意：prompt 已经包含了 <sent> 占位符，需要替换
                # 原 prompt: "<sent>\nSummarize above sentence in one word: "
                user_text = prompt.replace('<sent>', text_content)
                
                msgs = [
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "<emb>."}
                    ]}
                ]
                messages_list.append(msgs)
                
            else: # image embedding
                # 图像 Embedding
                img_item = batch[img_column][i]
                if isinstance(img_item, str):
                    # 是路径，需要打开
                    try:
                        img = Image.open(img_item).convert('RGB')
                    except:
                        # 如果加载失败，填充一个 dummy message 或跳过
                        print(f"Failed to load image: {img_item}")
                        continue
                else:
                    img = img_item
                
                # 获取图像尺寸
                w, h = img.size
                
                # 获取 bbox 信息（如果存在）
                bbox = batch.get('bbox', [None] * batch_size_current)[i]
                
                # 构造 messages
                # 格式: user: <image>\n prompt -> assistant: <emb>
                user_text = prompt # e.g. "Summarize above image in one word: "
                
                # 构造 image content
                image_content = {"type": "image", "image": img}
                
                # 如果有 bbox，添加归一化的 box 信息
                if bbox is not None:
                    # bbox 格式: [[x, y, w_box, h_box]]
                    if isinstance(bbox, list) and len(bbox) > 0:
                        # 取第一个 bbox
                        box_data = bbox[0]
                        x, y, w_box, h_box = box_data
                        # 归一化 bbox: [x1/w, y1/h, x2/w, y2/h]
                        normalized_box = [x/w, y/h, (x + w_box)/w, (y + h_box)/h]
                        image_content["box"] = normalized_box
                        image_content["box_op"] = BOX_OP

                        # import PIL.ImageDraw as ImageDraw
                        # draw = ImageDraw.Draw(img)

                        # draw.rectangle([x, y, x + w_box, y + h_box], outline="red", width=3)
                        # img.save("./xyz.jpg")
                        # breakpoint()
                
                msgs = [
                    {"role": "user", "content": [
                        image_content,
                        {"type": "text", "text": user_text}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "<emb>."}
                    ]}
                ]
                messages_list.append(msgs)
        
        # 调用模型生成 embedding
        # 注意：_run_model_batch 内部使用了循环，所以这里传入整个 list 没问题
        batch_embs = _run_model_batch(model, processor, device, messages_list)
        
        # Normalize
        batch_embs = F.normalize(batch_embs, dim=-1)
        
        # Gather (分布式)
        # batch_embs = accelerator.gather(batch_embs) # 若启用分布式
        
        embs.append(batch_embs.cpu().float())
        bar.update(1)
        
    bar.close()
    if len(embs) == 0:
        return torch.tensor([])
    return torch.cat(embs)


# --- SORCE-IR 评估逻辑 (保持原逻辑，调用新的 emb_data) ---

def sorce_recall_at_k_difficulty(scores, positive_pairs, k, difficulty='easy',
                                 total_difficulties=['easy', 'medium', 'hard']):
    # 保持原样，这是核心指标计算逻辑
    assert difficulty in ['easy', 'medium', 'hard']
    mod1, mod2 = scores.shape
    tmp_scores = scores.clone()
    tmp_positive_pairs = positive_pairs.clone()

    if mod1 < mod2:  # text to image retrieval
        for i in range(mod1):
            indices = torch.where(tmp_positive_pairs[i] == True)[0].chunk(len(total_difficulties))
            if difficulty == 'easy':
                if 'medium' in total_difficulties:
                    tmp_scores[i, indices[1]] = -float('inf') # Mask out
                if 'hard' in total_difficulties:
                    idx = indices[2] if 'medium' in total_difficulties else indices[1]
                    tmp_scores[i, idx] = -float('inf')
            elif difficulty == 'medium':
                if 'easy' in total_difficulties:
                    tmp_scores[i, indices[0]] = -float('inf')
                if 'hard' in total_difficulties:
                    idx = indices[2] if 'easy' in total_difficulties else indices[1]
                    tmp_scores[i, idx] = -float('inf')
            elif difficulty == 'hard':
                if 'easy' in total_difficulties:
                    tmp_scores[i, indices[0]] = -float('inf')
                if 'medium' in total_difficulties:
                    idx = indices[1] if 'easy' in total_difficulties else indices[0]
                    tmp_scores[i, idx] = -float('inf')

    topk_indices = torch.topk(tmp_scores, k, dim=1)[1]
    nb_positive = tmp_positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=mod2)
    positive_pairs_reshaped = tmp_positive_pairs.view(mod1, 1, mod2)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def sorce_ir(model, processor,
             img_prompt, text_prompt,
             data, device,
             batch_size=None, difficulty=['easy', 'medium', 'hard']):
    
    dataset = load_dataset("json", data_files=data)['train']
    dataset = dataset.rename_column('image', 'img')

    dirname = os.path.dirname(data)
    # 根据 SORCE 数据集的结构准备路径
    easy_dir = os.path.join(dirname, 'zoom_3x')
    medium_dir = os.path.join(dirname, 'zoom_2x')
    hard_dir = os.path.join(dirname, 'full_res')

    dataset_collection = []
    if 'easy' in difficulty:
        easy_dataset = dataset.map(lambda x: {'img': os.path.join(easy_dir, x['img'])}, num_proc=4)
        dataset_collection.append(easy_dataset)
    if 'medium' in difficulty:
        medium_dataset = dataset.map(lambda x: {'img': os.path.join(medium_dir, x['img'])}, num_proc=4)
        dataset_collection.append(medium_dataset)
    if 'hard' in difficulty:
        hard_dataset = dataset.map(lambda x: {'img': os.path.join(hard_dir, x['img'])}, num_proc=4)
        dataset_collection.append(hard_dataset)

    all_dataset = concatenate_datasets(dataset_collection)

    bsz = 4
    if batch_size is not None:
        bsz = batch_size

    with torch.no_grad():
        # 使用新的 emb_data 函数
        img_embs = emb_data(model, processor, all_dataset, device, emb_type='image', prompt=img_prompt, bsz=bsz)
        text_embs = emb_data(model, processor, dataset, device, emb_type='text', prompt=text_prompt, bsz=bsz)

    dataset_multiples = len(difficulty)
    image_text_index = [i for i in range(text_embs.shape[0])] * dataset_multiples

    assert text_embs.isnan().sum().item() == 0
    assert img_embs.isnan().sum().item() == 0

    scores = img_embs @ text_embs.t()
    
    print(f'Image embs shape: {img_embs.shape}')
    print(f'Text embs shape: {text_embs.shape}')
    print(f'Scores shape: {scores.shape}')

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), image_text_index] = True

    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size_eval = 64
    for recall_k in recall_k_list:
        for level in difficulty:
            metrics[f"[{level}] image_retrieval_recall@{recall_k}"] = (
                    batchify(sorce_recall_at_k_difficulty, scores.T, positive_pairs.T, batch_size_eval, device,
                             k=recall_k, difficulty=level, total_difficulties=difficulty) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores, positive_pairs, batch_size_eval, device,
                         k=recall_k) > 0).float().mean().item()

    return metrics


# --- Main (入口整合) ---

def main(
        model_id: str = None,
        original_model_id: str = None,
        batch_size: int = 4,
        data: str = None,
        difficulty: str = None,
):
    global emb_data_func
    
    # 目前暂不支持 EXTRA_PROMPTS (FGOVD 逻辑没有这部分)
    emb_data_func = emb_data

    device = accelerator.device

    # 初始化模型
    model, processor = init_model_and_transform(model_id, original_model_id)
    model.to(device)

    from datasets import disable_caching
    disable_caching()
    
    # 数据集路径
    datasets = ["/mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/sorce-1k/dataset.jsonl"]
    difficulty = ['hard'] # 'easy', 'medium', 

    if data:
        datasets = data.split(',')

    all_results = []
    for data_item in datasets:
        if 'sorce' in data_item:
            # 根据模型构造 Prompt
            # 因为 Qwen2VLRet 对应的是 chat template，我们不需要像 LLaVA 那样手动拼 [INST]
            # prompt 需要符合 Qwen 的格式或通用格式，具体由 processor.apply_chat_template 决定
            # 这里简单使用通用 Prompt，由 emb_data 内部构造 message
            img_prompt = "\nSummarize above image in one word: "
            text_prompt = "<sent>\nSummarize above sentence in one word: "
            
            metrics = sorce_ir(model, processor, img_prompt, text_prompt, data_item, device,
                               batch_size, difficulty=difficulty)
        else:
            # 可扩展支持 flickr30k 等
            raise ValueError(f"Dataset {data_item} not supported yet.")

        # if accelerator.is_main_process:
        #     print(metrics)
        #     os.makedirs("results", exist_ok=True)
        #     checkpoint_name = model_id.replace('/', '_') + '_results.txt'
        #     checkpoint_name = os.path.join("results", checkpoint_name)
        #     all_results.append(log_to_file(data_item, metrics, checkpoint_name, difficulty=difficulty))
        log_to_file(data_item, metrics, "abc", difficulty=difficulty)

    if accelerator.is_main_process:
        print('\n'.join(all_results))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
