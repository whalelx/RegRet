#!/bin/bash

SOURCE_DIR="/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data/SAM/images2"
TARGET_DIR="/mnt/tidal-alsh01/dataset/mmeb/describe-anything-data/SAM/images"

find "$SOURCE_DIR" -name "00000[0-9][0-9][0-9].tar" -exec sh -c '
    for file; do
        basename=$(basename "$file")
        # 提取数字部分
        num=$(echo "$basename" | grep -o "[0-9][0-9][0-9]\.tar$" | grep -o "^[0-9][0-9][0-9]")
        if [ "$num" -ge 0 ] && [ "$num" -le 100 ]; then
            mv "$file" "'"$TARGET_DIR"'/"
            echo "已移动: $basename"
        fi
    done
' sh {} +

cp $SOURCE_DIR/*.json $TARGET_DIR

echo "批量移动完成！"
