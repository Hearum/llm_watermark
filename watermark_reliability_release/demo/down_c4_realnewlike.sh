#!/bin/bash

# 设置基础 URL
BASE_URL="https://hf-mirror.com/datasets/allenai/c4/resolve/main/realnewslike"

# 训练集文件编号范围
TRAIN_START_INDEX=0
TRAIN_END_INDEX=511  # 512 个文件 (00000 到 00511)

# 验证集文件编号范围
VALID_START_INDEX=0
VALID_END_INDEX=0  # 只有一个文件 (00000-of-00001)

# 设置文件格式
FILE_EXTENSION=".json.gz?download=true"

# 设置并发下载数
NUM_PARALLEL=5

# 创建下载目录
DOWNLOAD_DIR="./downloads"
mkdir -p "$DOWNLOAD_DIR"

# 下载函数
download_file() {
    local FILE_TYPE=$1
    local FILE_INDEX=$(printf "%05d" $2)
    local TOTAL_FILES=$3
    local FILE_URL="${BASE_URL}/${FILE_TYPE}.${FILE_INDEX}-of-${TOTAL_FILES}${FILE_EXTENSION}"
    local OUTPUT_FILE="${DOWNLOAD_DIR}/${FILE_TYPE}.${FILE_INDEX}-of-${TOTAL_FILES}.json.gz"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ 文件已存在，跳过: $OUTPUT_FILE"
    else
        echo "⬇️  开始下载: $FILE_URL"
        wget --content-disposition -c "$FILE_URL" -O "$OUTPUT_FILE"
        if [ $? -eq 0 ]; then
            echo "✅ 下载完成: $OUTPUT_FILE"
        else
            echo "❌ 下载失败: $FILE_URL"
        fi
    fi
}

# 允许 `download_file` 被 `xargs` 并行执行
export -f download_file
export BASE_URL FILE_EXTENSION DOWNLOAD_DIR

# 下载训练集
seq $TRAIN_START_INDEX $TRAIN_END_INDEX | xargs -P $NUM_PARALLEL -I {} bash -c 'download_file "c4-train" "$@" "00512"' _ {}

# 下载验证集
seq $VALID_START_INDEX $VALID_END_INDEX | xargs -P $NUM_PARALLEL -I {} bash -c 'download_file "c4-validation" "$@" "00001"' _ {}

echo "🎉 所有下载任务完成！"
