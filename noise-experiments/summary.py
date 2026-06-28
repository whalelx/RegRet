import subprocess
import re
import os

# 定义目标参数范围
target_scales = [0.02, 0.05, 0.08, 0.12, 0.15, 0.30]
# 初始化两个独立的列表
scale_averages = []
translate_averages = []

def get_average_from_script(filename):
    """调用 print.py 并提取平均值"""
    try:
        # 假设调用方式：python print.py <filename>
        cmd = ["python", "print.py", filename]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # 使用正则提取浮点数 (例如 85.6)
        numbers = re.findall(r"\d+\.?\d*", output)
        if numbers:
            return float(numbers[-1])
        return None
    except Exception:
        return None

# 主循环：按顺序填充两个列表
for scale in target_scales:
    scale_str = f"{scale:.2f}"
    
    # 1. 处理 scale 文件
    scale_file = f"scale_{scale_str}_0.08_results.txt"
    if os.path.exists(scale_file):
        val = get_average_from_script(scale_file)
        if val is not None:
            scale_averages.append(val)
            print(f"已添加至 scale 列表：{scale_file} -> {val}")
            
    # 2. 处理 translate 文件
    translate_file = f"translate_{scale_str}_0.08_results.txt"
    if os.path.exists(translate_file):
        val = get_average_from_script(translate_file)
        if val is not None:
            translate_averages.append(val)
            print(f"已添加至 translate 列表：{translate_file} -> {val}")

# 输出最终结果
print("\n最终生成的两个列表:")
print(f"scale_averages: {scale_averages}")
print(f"translate_averages: {translate_averages}")
