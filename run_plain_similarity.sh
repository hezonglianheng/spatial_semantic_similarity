#!/bin/bash
# =============================================================================
# 运行 plain_similarity.py 的 bash 脚本
# =============================================================================
# 用法:
#   ./run_plain_similarity.sh [选项]
#
# 示例:
#   ./run_plain_similarity.sh
#   ./run_plain_similarity.sh -m bert-base-chinese -a bert-chinese
#   ./run_plain_similarity.sh -d my_data.json -m /path/to/model -a my_model -o ./output
# =============================================================================

set -euo pipefail

# ---------- 默认参数 ----------
DATA_FILE="spatial_dataset.json"
MODEL_NAME_OR_PATH="bert-base-chinese"
MODEL_ALIAS=""
OUTPUT_DIR="./output"

# ---------- 脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 帮助信息 ----------
usage() {
    cat << EOF
用法: $0 [选项]

选项:
  -d, --data-file PATH          数据文件路径 (默认: ${DATA_FILE})
  -m, --model PATH              模型名称或路径 (默认: ${MODEL_NAME_OR_PATH})
  -a, --alias NAME              模型别名, 用于输出文件命名 (默认: 自动从模型路径提取)
  -o, --output-dir DIR          输出目录 (默认: ${OUTPUT_DIR})
  -h, --help                    显示此帮助信息

示例:
  $0
  $0 -m bert-base-chinese -a bert-chinese
  $0 -d my_data.json -m /path/to/model -a my_model -o ./results
EOF
    exit 0
}

# ---------- 解析命令行参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data-file)
            DATA_FILE="$2"; shift 2 ;;
        -m|--model)
            MODEL_NAME_OR_PATH="$2"; shift 2 ;;
        -a|--alias)
            MODEL_ALIAS="$2"; shift 2 ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "[错误] 未知选项: $1"
            usage ;;
    esac
done

# ---------- 自动生成 MODEL_ALIAS ----------
if [[ -z "${MODEL_ALIAS}" ]]; then
    # 从模型路径中提取最后一部分作为别名, 替换掉路径分隔符和非字母数字字符
    MODEL_ALIAS=$(basename "${MODEL_NAME_OR_PATH}" | sed 's/[^a-zA-Z0-9_-]/_/g')
fi

# ---------- 检查数据文件 ----------
if [[ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ]]; then
    echo "[错误] 未找到数据文件: ${SCRIPT_DIR}/${DATA_FILE}"
    exit 1
fi

# ---------- 创建输出目录 ----------
mkdir -p "${OUTPUT_DIR}"

# ---------- 打印运行信息 ----------
echo "============================================"
echo "  运行 plain_similarity.py"
echo "============================================"
echo "  数据文件:   ${DATA_FILE}"
echo "  模型路径:   ${MODEL_NAME_OR_PATH}"
echo "  模型别名:   ${MODEL_ALIAS}"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  Python:     $(which python)"
echo "============================================"
echo ""

# ---------- 运行 Python 脚本 ----------
cd "${SCRIPT_DIR}"

python plain_similarity.py \
    --data_file "${DATA_FILE}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --model_alias "${MODEL_ALIAS}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "[完成] 结果已保存到 ${OUTPUT_DIR}/"
