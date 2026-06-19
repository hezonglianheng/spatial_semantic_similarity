#!/bin/bash
# =============================================================================
# 批量运行 simple_similarity.py — 支持指定多个模型及其别名
# =============================================================================
# 用法:
#   ./run_simple_similarity_batch.sh [选项]
#
# 指定模型的方式 (优先级从高到低):
#   1. --models "model1::alias1 model2::alias2 ..."  (命令行, 空格分隔)
#   2. 直接编辑下方 MODELS 数组
#
# 示例:
#   ./run_simple_similarity_batch.sh
#   ./run_simple_similarity_batch.sh --models "bert-base-chinese::bert-chinese Qwen/Qwen2-0.5B::qwen2-0.5b"
#   ./run_simple_similarity_batch.sh --models "bert-base-chinese::bert-chinese" -d my_data.json -o ./results
# =============================================================================

set -euo pipefail

# ---------- 脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 默认参数 ----------
DATA_FILE="spatial_dataset.json"
OUTPUT_DIR="./output"

# ---------- 默认模型列表 (格式: "模型路径::别名" 用空格分隔) ----------
# 别名用于输出文件命名; 若省略别名则从路径自动提取
DEFAULT_MODELS=(
    "bert-base-chinese::bert-chinese"
    # "Qwen/Qwen2-0.5B::qwen2-0.5b"
    # "sentence-transformers/all-MiniLM-L6-v2::minilm-l6"
)

# ---------- 帮助信息 ----------
usage() {
    cat << EOF
用法: $0 [选项]

选项:
  --models "M1::A1 M2::A2 ..."   模型列表: 每个条目为 模型路径::别名, 空格分隔
                                  别名可选, 省略时自动从路径提取
  -d, --data-file PATH           数据文件路径 (默认: ${DATA_FILE})
  -o, --output-dir DIR           输出目录 (默认: ${OUTPUT_DIR})
  -h, --help                     显示此帮助信息

模型指定方式:
  - 命令行: --models "bert-base-chinese::bert-chinese /path/to/model::my_model"
  - 脚本内置: 编辑脚本中的 DEFAULT_MODELS 数组
  - 命令行参数会完全覆盖脚本内置的默认模型列表

示例:
  # 使用脚本内置的默认模型
  $0

  # 命令行指定多个模型
  $0 --models "bert-base-chinese::bert-chinese Qwen/Qwen2-0.5B::qwen2-0.5b"

  # 省略别名, 自动从路径提取
  $0 --models "bert-base-chinese sentence-transformers/all-MiniLM-L6-v2"

  # 结合其他参数
  $0 --models "bert-base-chinese::bert" -d my_data.json -o ./results
EOF
    exit 0
}

# ---------- 解析命令行参数 ----------
MODELS_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            MODELS_ARG="$2"; shift 2 ;;
        -d|--data-file)
            DATA_FILE="$2"; shift 2 ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "[错误] 未知选项: $1"
            usage ;;
    esac
done

# ---------- 确定最终模型列表 ----------
MODEL_ENTRIES=()
if [[ -n "${MODELS_ARG}" ]]; then
    # 命令行指定了模型列表, 按空格拆分
    read -ra MODEL_ENTRIES <<< "${MODELS_ARG}"
else
    # 使用脚本内置默认模型列表
    MODEL_ENTRIES=("${DEFAULT_MODELS[@]}")
fi

if [[ ${#MODEL_ENTRIES[@]} -eq 0 ]]; then
    echo "[错误] 未指定任何模型。请使用 --models 参数或编辑 DEFAULT_MODELS 数组。"
    exit 1
fi

# ---------- 检查数据文件 ----------
if [[ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ]]; then
    echo "[错误] 未找到数据文件: ${SCRIPT_DIR}/${DATA_FILE}"
    exit 1
fi

# ---------- 创建输出目录 ----------
mkdir -p "${OUTPUT_DIR}"

# ---------- 打印批量运行信息 ----------
echo "============================================"
echo "  批量运行 simple_similarity.py"
echo "============================================"
echo "  数据文件:   ${DATA_FILE}"
echo "  输出目录:   ${OUTPUT_DIR}"
echo "  Python:     $(which python)"
echo "  模型数量:   ${#MODEL_ENTRIES[@]}"
echo "============================================"
echo ""

# ---------- 解析单个模型条目 ----------
# 输入: "model_path::alias" 或 "model_path"
# 输出: 设置 MODEL_PATH, MODEL_ALIAS
parse_model_entry() {
    local entry="$1"
    if [[ "${entry}" == *"::"* ]]; then
        MODEL_PATH="${entry%%::*}"
        MODEL_ALIAS="${entry##*::}"
    else
        MODEL_PATH="${entry}"
        MODEL_ALIAS=""
    fi

    # 自动生成别名 (若未指定或为空)
    if [[ -z "${MODEL_ALIAS}" ]]; then
        MODEL_ALIAS=$(basename "${MODEL_PATH}" | sed 's/[^a-zA-Z0-9_-]/_/g')
    fi
}

# ---------- 逐个运行模型 ----------
TOTAL=${#MODEL_ENTRIES[@]}
CURRENT=0
FAILED_MODELS=()

for entry in "${MODEL_ENTRIES[@]}"; do
    CURRENT=$((CURRENT + 1))

    parse_model_entry "${entry}"

    echo ""
    echo "============================================"
    echo "  [${CURRENT}/${TOTAL}] 模型: ${MODEL_ALIAS}"
    echo "       路径: ${MODEL_PATH}"
    echo "============================================"
    echo ""

    cd "${SCRIPT_DIR}"

    if python simple_similarity.py \
        --data_file "${DATA_FILE}" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_alias "${MODEL_ALIAS}" \
        --output_dir "${OUTPUT_DIR}"; then
        echo ""
        echo "[完成] ${MODEL_ALIAS} → ${OUTPUT_DIR}/"
    else
        echo ""
        echo "[失败] ${MODEL_ALIAS} 运行出错, 继续下一个..."
        FAILED_MODELS+=("${MODEL_ALIAS}")
    fi
done

# ---------- 汇总 ----------
echo ""
echo "============================================"
echo "  批量运行结束"
echo "============================================"
echo "  总计: ${TOTAL}  成功: $((TOTAL - ${#FAILED_MODELS[@]}))  失败: ${#FAILED_MODELS[@]}"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "  失败列表: ${FAILED_MODELS[*]}"
fi
echo "  结果目录: ${OUTPUT_DIR}/"
echo "============================================"
