#!/usr/bin/env bash

# 1. 定义两个数组：old[i] → new[i]
old=(/mnt/tidal-alsh01/usr/liangxun/data/coco2017/ visual-spatial-reasoning/images)
new=(/mnt/tidal-alsh01/usr/liangxun/data/coco2017/train2017/ visual-spatial-reasoning/images/)

file="/mnt/tidal-alsh01/usr/liangxun/STAUG/LLaMA-Factory/data/Mix_OSD_VPT_large.json"

for ((i=0; i<${#old[@]}; i++)); do
  sed -i "s|${old[i]}|${new[i]}|g" "$file"
done

echo "替换完成：已在 ${file} 中将 ${#old[@]} 对字符串替换完成。"
