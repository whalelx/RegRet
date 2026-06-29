import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel
import requests
from io import BytesIO
import json
from tqdm import tqdm # For a nice progress bar
import os

# Default system message if not provided in the input file
DEFAULT_SYSTEM_MESSAGE = (
  "You are VL-Thinking 🤔, a helpful assistant with excellent reasoning ability. "
  "You should first think about the reasoning process and then provide the answer. "
  "Use <think>...</think> and <answer>...</answer> tags."
)

def batch_predict_from_sharegpt(
    input_sharegpt_path: str,
    output_jsonl_path: str,
    model_id: str = "remyxai/SpaceThinker-Qwen2.5VL-3B",
    batch_size: int = 4, # Adjust batch_size based on your GPU VRAM
    max_image_width: int = 512, # Max width to resize images to
    max_new_tokens: int = 1024
):
    """
    Processes a ShareGPT-like JSONL file in batches to generate predictions.

    Args:
        input_sharegpt_path (str): Path to the input JSONL file.
            Each line should be a JSON object with keys:
            "image_path" (str): Local path or URL to the image.
            "prompt" (str): The user's text prompt.
            "system_message" (str, optional): System message. Defaults to DEFAULT_SYSTEM_MESSAGE.
            "id" (str, optional): An identifier for the item.
        output_jsonl_path (str): Path to save the output JSONL file.
            Each line will be {"id": original_id (if provided), "predict": model_output}.
        model_id (str): Hugging Face model identifier.
        batch_size (int): Number of items to process in a single batch.
        max_image_width (int): Images wider than this will be resized.
        max_new_tokens (int): Maximum new tokens to generate.
    """

    print(f"Loading model and processor: {model_id}...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained( # Qwen2_5_VLForConditionalGeneration
            model_id,
            device_map="auto", # Automatically uses CUDA if available
            torch_dtype=torch.bfloat16 # or torch.float16 if bfloat16 not supported
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model/processor: {e}")
        return
    
    model.eval() # Set model to evaluation mode

    # Read all data from the input file
    all_input_data = []

    print(f"Reading input data from: {input_sharegpt_path}...")
    try:
        with open(input_sharegpt_path, 'r', encoding='utf-8') as f_in:
            all_input_data.extend(json.load(f_in))
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_sharegpt_path}")
        return
    
    if not all_input_data:
        print("No valid data found in input file.")
        return

    print(f"Starting batch processing for {len(all_input_data)} items with batch_size {batch_size}...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(all_input_data), batch_size), desc="Processing Batches"):
            batch_items = all_input_data[i:i+batch_size]
            
            current_batch_pil_images = []
            current_batch_chat_templates_for_processor = [] # List of 'chat' dicts for apply_chat_template
            original_ids_in_batch = []

            for item_idx, item in enumerate(batch_items):
                image_path = item.get("images")[0]
                prompt = item.get("messages")[0]["content"]
                item_id = item.get("index", f"item_{i+item_idx}") # Default ID if not provided
                original_ids_in_batch.append(item_id)

                system_message = item.get("system_message", DEFAULT_SYSTEM_MESSAGE)

                # Load and preprocess image
                try:
                    if image_path.startswith("http"):
                        response = requests.get(image_path, timeout=10)
                        response.raise_for_status() # Raise an exception for HTTP errors
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        if not os.path.exists(image_path):
                            print(f"Warning: Local image path not found: {image_path} for item id '{item_id}'. Skipping.")
                            original_ids_in_batch.pop() # Remove ID since we are skipping
                            continue
                        image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Could not load image {image_path} for item id '{item_id}': {e}. Skipping.")
                    original_ids_in_batch.pop() # Remove ID since we are skipping
                    continue

                if image.width > max_image_width:
                    ratio = image.height / image.width
                    image = image.resize((max_image_width, int(max_image_width * ratio)), Image.Resampling.LANCZOS)
                
                current_batch_pil_images.append(image)

                # Format input chat structure for this item
                # This structure is passed to apply_chat_template
                chat_for_item = [
                    {"role": "system", "content": [{"type": "text", "text": system_message}]},
                    {"role": "user", "content": [{"type": "image", "image": image}, # Pass the PIL image object
                                                {"type": "text", "text": prompt}]}
                ]
                current_batch_chat_templates_for_processor.append(chat_for_item)

            if not current_batch_pil_images: # If all items in batch were skipped
                continue

            # Apply chat template for each item in the batch
            # This creates a list of formatted text strings, one for each item
            try:
                batch_text_inputs_for_processor = [
                    processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
                    for chat_template in current_batch_chat_templates_for_processor
                ]
            except Exception as e:
                print(f"Error during apply_chat_template for batch starting at index {i}: {e}. Skipping batch.")
                # Write error for associated IDs
                for skipped_id in original_ids_in_batch: # only for those successfully loaded
                     f_out.write(json.dumps({"id": skipped_id, "predict": f"Error: apply_chat_template failed - {e}"}) + "\n")
                continue


            # Tokenize the batch of texts and images
            # The `processor` takes a list of texts and a list of images
            try:
                inputs = processor(
                    text=batch_text_inputs_for_processor,
                    images=current_batch_pil_images,
                    return_tensors="pt",
                    padding=True,  # Important for batching text
                    truncation=True # Ensure inputs are not too long
                ).to(model.device) # Move inputs to the same device as the model
            except Exception as e:
                print(f"Error during processor tokenization for batch starting at index {i}: {e}. Skipping batch.")
                for skipped_id in original_ids_in_batch:
                     f_out.write(json.dumps({"id": skipped_id, "predict": f"Error: processor tokenization failed - {e}"}) + "\n")
                continue

            # Generate response for the batch
            try:
                with torch.no_grad(): # Important for inference
                    generated_ids_batch = model.generate(**inputs, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"Error during model.generate for batch starting at index {i}: {e}. Skipping batch.")
                for skipped_id in original_ids_in_batch:
                     f_out.write(json.dumps({"id": skipped_id, "predict": f"Error: model generation failed - {e}"}) + "\n")
                continue
            
            # Decode batch responses
            try:
                num_prompt_tokens = inputs['input_ids'].shape[1]
                answers_ids_batch = []
                for j in range(generated_ids_batch.shape[0]):
                    # generated_ids_batch[j] 包含了 prompt tokens + answer tokens
                    # 我们要从 prompt tokens 之后开始取，即答案的 tokens
                    # 注意：某些模型或 generate 配置可能不完全复制输入，
                    # 但通常情况下，生成的序列以输入序列为前缀。
                    # 如果您的 generate 配置指定了只返回新生成的 token（如 LLaMA 的某些实现或特定参数），
                    # 那么这里的切片逻辑可能需要调整或省略。
                    # 但标准的 Hugging Face generate 行为是返回完整序列。
                    
                    # generated_ids_batch[j] 的长度可能大于 num_prompt_tokens
                    # 如果生成的token数量为0，answer_tokens 会是空列表
                    answer_tokens = generated_ids_batch[j][num_prompt_tokens:]
                    answers_ids_batch.append(answer_tokens)

                # 使用 batch_decode 解码只包含答案的 token IDs
                outputs_batch = processor.batch_decode(answers_ids_batch, skip_special_tokens=True)
                # outputs_batch = processor.batch_decode(generated_ids_batch, skip_special_tokens=True)
            except Exception as e:
                print(f"Error during processor.batch_decode for batch starting at index {i}: {e}. Skipping batch.")
                for skipped_id in original_ids_in_batch:
                     f_out.write(json.dumps({"id": skipped_id, "predict": f"Error: batch decode failed - {e}"}) + "\n")
                continue

            # Write outputs to file
            for item_id, output_text in zip(original_ids_in_batch, outputs_batch):
                f_out.write(json.dumps({"id": item_id, "predict": output_text}) + "\n")
                print(output_text)
            f_out.flush() # Ensure data is written to disk periodically

    print(f"Batch processing complete. Output saved to: {output_jsonl_path}")


if __name__ == '__main__':
    input_file = ""
    # 2. Define output file path
    output_file = "output_predictions.jsonl"

    # 3. Call the batch processing function
    # Adjust batch_size if you encounter VRAM issues. Start with a small batch_size like 1 or 2.
    batch_predict_from_sharegpt(
        input_sharegpt_path=input_file,
        output_jsonl_path=output_file,
        batch_size=2, # Example batch size
        max_new_tokens=512 # Shorter for faster testing
    )

    # 4. (Optional) Print contents of the output file
    print(f"\n--- Contents of {output_file} ---")
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out_display:
            for line in f_out_display:
                print(line.strip())
    except FileNotFoundError:
        print(f"Output file {output_file} not found.")

