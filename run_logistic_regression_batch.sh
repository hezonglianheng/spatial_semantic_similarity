#!/bin/bash
# =============================================================================
# 批量运行 logistic_regression.py — 对目录下所有相似度 CSV 逐一做逻辑回归
# =============================================================================
# 用法:
#   ./run_logistic_regression_batch.sh [选项]
#
# 会扫描 INPUT_DIR 下所有匹配模式 (*similarities*.csv / spatial_similarities_*.csv)
# 的相似度文件, 对每个文件调用 logistic_regression.py 进行分析。
#
# 模型别名自动从文件名中提取 (例如:
#   similarities_Qwen3.5-9B_prompt1_mean.csv  →  Qwen3.5-9B_prompt1_mean)
#
# 示例:
#   # 使用默认参数, 扫描 ./output/ 下所有相似度 CSV
#   ./run_logistic_regression_batch.sh
#
#   # 指定输入/输出目录及数据文件
#   ./run_logistic_regression_batch.sh -i ./output/spatial_word_embedding -o ./logistic_results
#
#   # 按层聚合分析 (mean)
#   ./run_logistic_regression_batch.sh --aggregate mean
#
#   # 只分析特定层
#   ./run_logistic_regression_batch.sh --layers 0 5 10 15 20
#
#   # 通过自定义 CSV 文件列表运行
#   ./run_logistic_regression_batch.sh --csv-files "output/a.csv output/b.csv"
#
#   # 自定义 CSV 模式 (用于匹配非标准命名)
#   ./run_logistic_regression_batch.sh --csv-pattern "my_sim_*.csv"
# =============================================================================

set -euo pipefail

# ---------- 脚本所在目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 默认参数 ----------
INPUT_DIR="./output"
DATA_FILE="spatial_info_annotation/spatial_dataset_deepseek-v4-pro_20260618-143434_modified.json"
OUTPUT_DIR="./logistic_output"
AGGREGATE=""            # 留空 = 逐层分析; 可选: mean, max, min
LAYERS=()               # 留空 = 全部分析; 示例: (0 5 10 15)

# ---------- CSV 发现模式 ----------
# 默认模式: 匹配常见的相似度文件命名
DEFAULT_CSV_PATTERNS=(
    "*similarities*.csv"
    "spatial_similarities_*.csv"
)

# ---------- 帮助信息 ----------
usage() {
    cat << EOF
用法: $0 [选项]

选项:
  -i, --input-dir DIR            相似度 CSV 文件所在目录 (默认: ${INPUT_DIR})
  -d, --data-file PATH           数据集 JSON 文件路径 (默认: ${DATA_FILE})
  -o, --output-dir DIR           逻辑回归结果输出目录 (默认: ${OUTPUT_DIR})
  --csv-files "F1 F2 ..."        显式指定要分析的 CSV 文件列表 (空格分隔)
                                 指定后不再扫描 INPUT_DIR
  --csv-pattern "PAT"            自定义 CSV 文件匹配模式 (覆盖默认模式)
                                 可多次指定
  --aggregate MODE               跨层聚合方式: mean, max, min (默认: 逐层分析)
  --layers L1 L2 ...             指定分析的层索引 (默认: 全部层)
  -h, --help                     显示此帮助信息

示例:
  # 使用默认参数
  $0

  # 扫描指定目录
  $0 -i ./output/spatial_word_embedding -o ./logistic_results

  # 显式指定 CSV 文件
  $0 --csv-files "output/similarities_Qwen_p1_mean.csv output/similarities_llama_p1_mean.csv"

  # 按层均值聚合
  $0 --aggregate mean

  # 只分析第 0, 5, 10, 15 层
  $0 --layers 0 5 10 15

  # 自定义文件名模式
  $0 -i ./results --csv-pattern "my_sim_*.csv"
EOF
    exit 0
}

# ---------- 解析命令行参数 ----------
CSV_FILES_ARG=""
CSV_PATTERNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input-dir)
            INPUT_DIR="$2"; shift 2 ;;
        -d|--data-file)
            DATA_FILE="$2"; shift 2 ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --csv-files)
            CSV_FILES_ARG="$2"; shift 2 ;;
        --csv-pattern)
            CSV_PATTERNS+=("$2"); shift 2 ;;
        --aggregate)
            AGGREGATE="$2"; shift 2 ;;
        --layers)
            shift           # 吃掉 --layers
            LAYERS=()
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                LAYERS+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage ;;
        *)
            echo "[错误] 未知选项: $1"
            usage ;;
    esac
done

# ---------- 确定 CSV 匹配模式 ----------
if [[ ${#CSV_PATTERNS[@]} -eq 0 ]]; then
    CSV_PATTERNS=("${DEFAULT_CSV_PATTERNS[@]}")
fi

# ---------- 确定 CSV 文件列表 ----------
CSV_FILES=()

if [[ -n "${CSV_FILES_ARG}" ]]; then
    # 显式指定
    read -ra CSV_FILES <<< "${CSV_FILES_ARG}"
else
    # 扫描目录
    if [[ ! -d "${SCRIPT_DIR}/${INPUT_DIR}" ]]; then
        echo "[错误] 输入目录不存在: ${SCRIPT_DIR}/${INPUT_DIR}"
        exit 1
    fi

    for pattern in "${CSV_PATTERNS[@]}"; do
        while IFS= read -r f; do
            CSV_FILES+=("$f")
        done < <(find "${SCRIPT_DIR}/${INPUT_DIR}" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | sort)
    done
fi

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "[错误] 未找到任何相似度 CSV 文件。"
    echo "  搜索目录: ${SCRIPT_DIR}/${INPUT_DIR}"
    echo "  匹配模式: ${CSV_PATTERNS[*]}"
    echo "  可使用 --csv-files 显式指定文件, 或 --csv-pattern 调整匹配模式。"
    exit 1
fi

# ---------- 检查数据文件 ----------
if [[ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ]]; then
    echo "[错误] 未找到数据文件: ${SCRIPT_DIR}/${DATA_FILE}"
    exit 1
fi

# ---------- 创建输出目录 ----------
mkdir -p "${OUTPUT_DIR}"

# ---------- 构建 python 命令的公共参数 ----------
PYTHON_ARGS=(
    --data_file "${DATA_FILE}"
)

if [[ -n "${AGGREGATE}" ]]; then
    PYTHON_ARGS+=(--aggregate "${AGGREGATE}")
fi

if [[ ${#LAYERS[@]} -gt 0 ]]; then
    PYTHON_ARGS+=(--layers "${LAYERS[@]}")
fi

# ---------- 打印批量运行信息 ----------
echo "============================================"
echo "  批量运行 logistic_regression.py"
echo "============================================"
echo "  数据文件:     ${DATA_FILE}"
echo "  输出目录:     ${OUTPUT_DIR}"
echo "  CSV 文件数:   ${#CSV_FILES[@]}"
if [[ -n "${AGGREGATE}" ]]; then
    echo "  聚合方式:     ${AGGREGATE}"
else
    echo "  分析方式:     逐层分析"
fi
if [[ ${#LAYERS[@]} -gt 0 ]]; then
    echo "  指定层:       ${LAYERS[*]}"
else
    echo "  指定层:       全部"
fi
echo "  Python:       $(which python)"
echo "============================================"
echo ""
echo "相似度 CSV 文件列表:"
for f in "${CSV_FILES[@]}"; do
    echo "  - $(basename "$f")"
done
echo ""

# ---------- 从 CSV 文件名提取模型别名 ----------
# 输入: 文件路径 (如 output/similarities_Qwen3.5-9B_prompt1_mean.csv)
# 输出: 别名 (如 Qwen3.5-9B_prompt1_mean)
extract_alias() {
    local filepath="$1"
    local basename
    basename=$(basename "${filepath}" .csv)

    # 去掉常见前缀
    local alias="${basename}"
    alias="${alias#similarities_}"
    alias="${alias#spatial_similarities_}"

    echo "${alias}"
}

# ---------- 逐个分析 CSV 文件 ----------
TOTAL=${#CSV_FILES[@]}
CURRENT=0
FAILED_FILES=()

for csv_file in "${CSV_FILES[@]}"; do
    CURRENT=$((CURRENT + 1))

    MODEL_ALIAS=$(extract_alias "${csv_file}")
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_ALIAS}"
    mkdir -p "${MODEL_OUTPUT_DIR}"

    echo ""
    echo "============================================"
    echo "  [${CURRENT}/${TOTAL}] 别名: ${MODEL_ALIAS}"
    echo "       文件: $(basename "${csv_file}")"
    echo "       输出: ${MODEL_OUTPUT_DIR}"
    echo "============================================"
    echo ""

    cd "${SCRIPT_DIR}"

    if python logistic_regression.py \
        --similarities "${csv_file}" \
        --model_alias "${MODEL_ALIAS}" \
        --output_dir "${MODEL_OUTPUT_DIR}" \
        "${PYTHON_ARGS[@]}"; then
        echo ""
        echo "[完成] ${MODEL_ALIAS} → ${MODEL_OUTPUT_DIR}/"
    else
        echo ""
        echo "[失败] ${MODEL_ALIAS} (${csv_file}) 运行出错, 继续下一个..."
        FAILED_FILES+=("${MODEL_ALIAS}")
    fi
done

# ---------- 汇总 ----------
echo ""
echo "============================================"
echo "  批量运行结束"
echo "============================================"
echo "  总计: ${TOTAL}  成功: $((TOTAL - ${#FAILED_FILES[@]}))  失败: ${#FAILED_FILES[@]}"
if [[ ${#FAILED_FILES[@]} -gt 0 ]]; then
    echo "  失败列表:"
    for f in "${FAILED_FILES[@]}"; do
        echo "    - ${f}"
    done
fi
echo "  结果目录: ${OUTPUT_DIR}/<模型别名>/"
echo ""
echo "  结果文件:"
find "${OUTPUT_DIR}" -maxdepth 2 -name "logistic_summary_*.csv" -type f 2>/dev/null | sort | while read f; do
    echo "    ${f#${OUTPUT_DIR}/}"
done
if [[ -z $(find "${OUTPUT_DIR}" -maxdepth 2 -name "logistic_summary_*.csv" -type f 2>/dev/null) ]]; then
    echo "    (暂无)"
fi
echo "============================================"
