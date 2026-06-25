# from transformers import  AutoProcessor
# from transformers import  Qwen3VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info
# from models.qwen3_vl import Qwen3VLRetForConditionalGeneration
# # You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
# messages = [
#     # Image
#     ## Local file path
#     [{"role": "user", "content": [
#         {"type": "image", "image": "./demo.jpeg"}, 
#         {"type": "text", "text": "Describe this image."}
#     ]}],
#     ## Image URL
#     # [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
# ]

# model_path = "/mnt/tidalfs-hssh01/dataset/mmeb/Qwen3-VL-2B-Instruct"

# processor = AutoProcessor.from_pretrained(model_path)
# model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")

# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

# if videos is not None:
#     videos, video_metadatas = zip(*videos)
#     videos, video_metadatas = list(videos), list(video_metadatas)
# else:
#     video_metadatas = None

# inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
# inputs = inputs.to(model.device)

# generated_ids = model.generate(**inputs)
# print(generated_ids)

# output_text = processor.batch_decode(
#     generated_ids,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=False,
# )

# print(output_text)


from transformers import  AutoProcessor, AutoTokenizer
from transformers import  Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from models.qwen3_vl import Qwen3VLRetForConditionalGeneration
from collators.qwen3_vl_2b import Qwen3VL2BDataCollator
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [
        {"type": "image", "image": "./demo.jpeg"}, 
        {"type": "text", "text": "Describe this image."}
    ]}],
    ## Image URL
    # [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
]

model_path = "/mnt/tidalfs-hssh01/dataset/mmeb/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")

collactor = Qwen3VL2BDataCollator(
    tokenizer=tokenizer,
    processor=processor
)
inputs=collactor(messages)




text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
inputs = inputs.to(model.device)


generated_ids = model.generate(**inputs)

pred_ids = torch.argmax(output.logits, dim=-1)  


print(generated_ids)

output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(output_text)

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
a = torch.load("./debug_logits.pt")
labels=a["labels"]
preds=a["pred_ids"]
input_ids=a["input_ids"]
model_path = "/mnt/tidalfs-hssh01/dataset/mmeb/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
i=0
# print(processor.batch_decode(labels[i][:79],skip_special_tokens=True,clean_up_tokenization_spaces=False,))
print(processor.batch_decode(preds[i],skip_special_tokens=True,clean_up_tokenization_spaces=False,))
print(processor.batch_decode(input_ids[i],skip_special_tokens=True,clean_up_tokenization_spaces=False,))
print(processor.batch_decode(pred_ids[0],skip_special_tokens=True,clean_up_tokenization_spaces=False,))
