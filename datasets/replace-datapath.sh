#!/usr/bin/env bash

# 1. 定义两个数组：old[i] → new[i]
old=(/root/dataset/OpenImage/ /root/dataset/COCO/train2017/ /root/dataset/VSR/images/ /root/dataset/GQA/)
new=(/data/openimagesv7/ /mnt/tidal-alsh01/usr/liangxun/data/coco2017/train2017/ /mnt/tidal-alsh01/usr/liangxun/data/visual-spatial-reasoning/images/ /data/GQA/)

file="/mnt/tidal-alsh01/usr/liangxun/STAUG/LLaMA-Factory/data/mix-vpt-spatial.json"
file="/mnt/tidal-alsh01/usr/liangxun/STAUG/LLaMA-Factory/data/Mix_OSD_VPT_Cot450k.json"
file="/mnt/tidal-alsh01/usr/liangxun/STAUG/datasets/VSR_region_test.json"
file="/mnt/tidal-alsh01/usr/liangxun/STAUG/LLaMA-Factory/data/xyz.json"

for ((i=0; i<${#old[@]}; i++)); do
  sed -i "s|${old[i]}|${new[i]}|g" "$file"
done

old=(train_0 train_1 train_2 train_3 train_4 train_5)
new=(train train train train train train)

for ((i=0; i<${#old[@]}; i++)); do
  sed -i "s|${old[i]}|${new[i]}|g" "$file"
done

echo "替换完成：已在 ${file} 中将 ${#old[@]} 对字符串替换完成。"
