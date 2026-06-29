#!/bin/bash

# --- Configuration ---
# List of original filenames to process
# 您可以在这里添加或修改文件名列表
FILENAMES=(
    # "qspatial.json"
    # "VSR_region_test.json"
    # "srgpt_qualitive.json"
    # "whatsup.json"
    "robospatial-compatibility.json"
    "robospatial-configuration.json"
)

SEARCH_STRING="Identify the regions that can help you answer the question, and then answer the question."
REPLACE_STRING=""
NEW_FILENAME_SUFFIX="_newprompts" # 添加到基本文件名的后缀
# --- End Configuration ---

echo "开始执行脚本..."

# 检查 FILENAMES 数组是否为空
if [ ${#FILENAMES[@]} -eq 0 ]; then
    echo "错误：FILENAMES 列表为空。请在脚本中指定要处理的文件。"
    exit 1
fi

# 遍历所有指定的原始文件名
for original_file in "${FILENAMES[@]}"; do
    echo "----------------------------------------"
    echo "正在处理原始文件: $original_file"

    # 1. 检查原始文件是否存在
    if [ ! -f "$original_file" ]; then
        echo "警告: 原始文件 '$original_file' 未找到。已跳过。"
        continue # 跳到列表中的下一个文件
    fi

    # 2. 构建新的文件名
    # 提取目录路径 (如果有)
    dir_path=$(dirname "$original_file")
    # 提取不带目录的基本文件名
    base_filename=$(basename "$original_file")

    # 提取不带扩展名的文件名部分
    filename_no_ext="${base_filename%.*}"
    # 提取扩展名
    extension="${base_filename##*.}"

    new_base_filename=""
    # 处理没有扩展名或点文件的情况，确保后缀正确添加
    if [[ "$base_filename" == "$extension" ]] && [[ "$base_filename" != .* ]]; then
        # 文件没有扩展名 (例如 "myfile")
        new_base_filename="${base_filename}${NEW_FILENAME_SUFFIX}"
    elif [[ "$base_filename" == .* ]] && [[ "$filename_no_ext" == "" ]]; then
        # 点文件且没有其他点 (例如 ".bashrc")
        # filename_no_ext 会是空，base_filename 是 ".bashrc", extension 是 "bashrc"
        # 或者文件就是 "." 开头，没有后续扩展名，如 ".config"
        # 这种情况下，我们希望是 .config_newprompts
        new_base_filename="${base_filename}${NEW_FILENAME_SUFFIX}"
    else
        # 文件有扩展名 (例如 "file.json" 或 ".hidden.config")
        new_base_filename="${filename_no_ext}${NEW_FILENAME_SUFFIX}.${extension}"
    fi

    # 如果原始文件在子目录中，则将新文件也放在该子目录
    if [ "$dir_path" == "." ]; then
        new_file="$new_base_filename"
    else
        new_file="${dir_path}/${new_base_filename}"
    fi

    echo "将创建/覆盖新文件: $new_file"

    # 3. 执行替换并将输出重定向到新文件 [citation:4][citation:5]
    # 使用 sed 从 original_file 读取，并将处理后的输出写入 new_file
    # sed 的 's' 命令的 'g' 标志确保替换行内所有匹配项
    sed "s|${SEARCH_STRING}|${REPLACE_STRING}|g" "$original_file" > "$new_file"

    # 检查 sed 命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "成功创建并处理了 '$new_file'。"
    else
        echo "错误: 处理 '$original_file' 并写入 '$new_file' 时失败。"
        # 可选：如果 sed 失败，清理部分创建的新文件
        # rm -f "$new_file"
        # echo "已清理可能未完成的文件: $new_file"
    fi
done

echo "----------------------------------------"
echo "脚本执行完毕。"
