#!/usr/bin/env python3
"""
简化的DAM API客户端
实现输入图片和bounding box，生成对box区域的详细描述
"""

import argparse
import base64
import numpy as np
from PIL import Image
from openai import OpenAI
from io import BytesIO
from transformers import SamModel, SamProcessor
import torch
import requests
import json


class SimpleDAMClient:
    def __init__(self, server_url="http://localhost:8000", model_name="describe_anything_model"):
        """
        初始化DAM客户端
        
        Args:
            server_url: DAM服务器地址
            model_name: 模型名称
        """
        self.server_url = server_url
        self.model_name = model_name
        self.client = OpenAI(api_key="api_key", base_url=server_url)
        
        # 初始化SAM模型用于生成mask
        print("正在加载SAM模型...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
        self.device = device
        print(f"SAM模型已加载到设备: {device}")
    
    def generate_mask_from_box(self, image, box):
        """
        从bounding box生成mask
        
        Args:
            image: PIL Image对象
            box: bounding box [x1, y1, x2, y2]
        
        Returns:
            numpy array mask
        """
        # 确保box格式正确
        if len(box) != 4:
            raise ValueError("Bounding box must have 4 values: [x1, y1, x2, y2]")
        
        input_boxes = [[box]]  # SAM需要嵌套列表格式
        
        inputs = self.sam_processor(
            image, 
            input_boxes=input_boxes, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        
        scores = outputs.iou_scores[0, 0]
        mask_selection_index = scores.argmax()
        mask_np = masks[mask_selection_index].numpy()
        
        return mask_np
    
    def create_rgba_image(self, image, mask_np):
        """
        创建RGBA图像（RGB + Alpha mask）
        
        Args:
            image: PIL Image对象
            mask_np: numpy array mask
        
        Returns:
            RGBA PIL Image
        """
        mask = Image.fromarray((mask_np * 255).astype(np.uint8))
        rgba_image = Image.merge("RGBA", image.split() + (mask,))
        return rgba_image
    
    def image_to_base64(self, rgba_image):
        """
        将RGBA图像转换为base64编码
        
        Args:
            rgba_image: RGBA PIL Image
        
        Returns:
            base64编码的图像字符串
        """
        buffered = BytesIO()
        rgba_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    
    def describe_region(self, image, box, query="\nDescribe the masked region in detail.", stream=False):
        """
        对图像中指定区域进行描述
        
        Args:
            box: bounding box [x1, y1, x2, y2]
            query: 查询文本
            stream: 是否使用流式输出
        
        Returns:
            描述文本
        """
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            print(f"图像尺寸: {image.size}")
        else:
            pass

        # 验证bounding box
        width, height = image.size
        x1, y1, x2, y2 = box
        x1*=width
        y1*=height
        x2*=width
        y2*=height
        box = [x1, y1, x2, y2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise ValueError(f"无效的bounding box {box}，图像尺寸为 {image.size}")
        
        # 生成mask
        print("正在生成mask...")
        mask_np = self.generate_mask_from_box(image, box)
        
        # 创建RGBA图像
        rgba_image = self.create_rgba_image(image, mask_np)
        
        # 转换为base64
        img_base64 = self.image_to_base64(rgba_image)
        
        # 构建请求
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_base64}
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
        ]
        
        print("正在发送请求到DAM服务器...")
        
        # 发送请求
        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.2,
                    top_p=0.9,
                    stream=True
                )
                
                print("描述结果（流式输出）:")
                result = ""
                for chunk in response:
                    if chunk.choices[0].delta and chunk.choices[0].delta.content:
                        for content_piece in chunk.choices[0].delta.content:
                            if content_piece['type'] == 'text':
                                result += content_piece['text']
                                print(content_piece['text'], end='', flush=True)
                print()  # 换行
                return result
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.2,
                    top_p=0.9
                )
                result = response.choices[0].message.content
                print(f"描述结果: {result}")
                return result
                
        except Exception as e:
            raise RuntimeError(f"API请求失败: {e}")
    
    def test_connection(self):
        """测试与服务器的连接"""
        try:
            # response = requests.get(f"{self.server_url.rstrip('/')}/")
            response = requests.get(f"{self.server_url.rstrip('/')}/")
            print(f"服务器连接测试: {response.status_code}")
            return True
        except Exception as e:
            print(f"无法连接到服务器 {self.server_url}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="DAM API客户端 - 输入图片和bounding box生成描述")
    parser.add_argument("--image", type=str, required=True, help="图像文件路径")
    parser.add_argument("--box", type=str, required=True, 
                       help="Bounding box，格式: 'x1,y1,x2,y2' 例如: '100,100,300,300'")
    parser.add_argument("--query", type=str, default="请详细描述这个区域的内容", 
                       help="查询文本")
    parser.add_argument("--server", type=str, default="http://localhost:8000", 
                       help="DAM服务器地址")
    parser.add_argument("--model", type=str, default="describe_anything_model", 
                       help="模型名称")
    parser.add_argument("--stream", action="store_true", help="使用流式输出")
    
    args = parser.parse_args()
    
    # 解析bounding box
    try:
        box = [float(x.strip()) for x in args.box.split(',')]
        if len(box) != 4:
            raise ValueError("Bounding box必须包含4个数值")
    except Exception as e:
        print(f"无效的bounding box格式: {e}")
        print("正确格式示例: --box '100,100,300,300'")
        return
    
    # 创建客户端
    try:
        client = SimpleDAMClient(args.server, args.model)
        
        # 测试连接
        if not client.test_connection():
            print("请确保DAM服务器正在运行")
            return
        
        # 生成描述
        description = client.describe_region(
            image=args.image,
            box=box,
            query=args.query,
            stream=args.stream
        )
        
        print(f"\n最终描述结果:\n{description}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
    # python dataset/xhs_datatools/dam_client.py --box "0.50237, 0.2811, 0.99541, 0.87666" --image "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/goods_data/from_20250401_to_20250407/images/2emmkfymylq6sq6gayvwg_notes_pre_post_1040g3k831cc22qoe0c305p2vo4mk5815dgogjp0.jpg" --query "What is the background's color on the left part of the region?"  
    # Answer:The background on the left part of the image is black.
    # python dataset/xhs_datatools/dam_client.py --box "0.50237, 0.2811, 0.99541, 0.87666" --image "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/goods_data/from_20250401_to_20250407/images/2emmkfymylq6sq6gayvwg_notes_pre_post_1040g3k831cc22qoe0c305p2vo4mk5815dgogjp0.jpg" --query "How many people are there in the image" 
    # Answer: There is one people in the image.(但是如果我问in total， 就会说两个)
    # python dataset/xhs_datatools/dam_client.py --box "0.23, 0.42, 0.73, 0.71" --image "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/goods_data/from_20250401_to_20250407/images/2emnk2bvvutyhcztn864g_active_search_1040g0mg31fs20v6rge5g5nddhh908p0g43hk6jo.jpg" --query "What is the cartoon charactor on the right of the region, pasted on the wall"
    # A cartoon character with a brown crossbody bag, wearing a white long-sleeved shirt with a round neckline.
    # python dataset/xhs_datatools/dam_client.py --box "0.23, 0.42, 0.73, 0.71" --image "/mnt/tidal-alsh01/dataset/mmeb/xhs_data/goods_data/from_20250401_to_20250407/images/2emnk2bvvutyhcztn864g_active_search_1040g0mg31fs20v6rge5g5nddhh908p0g43hk6jo.jpg" --query "What is on the right of the region? Describe that in detail"
    # The cartoon character on the right of the image is a red, cartoonish face with large, round eyes and a small, smiling mouth.