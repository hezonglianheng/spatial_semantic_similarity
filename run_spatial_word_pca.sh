#!/usr/bin/env bash
# ============================================================================
# 批量运行 spatial_word_pca.py —— 对多个模型的所有层做空间词嵌入 PCA 分析
# ============================================================================
#
# 用法:
#   1. 修改下方 MODELS 数组，填入「模型路径|模型别名」
#   2. ./run_spatial_word_pca.sh
#
#   也可通过命令行传入模型（覆盖 MODELS 数组）:
#   ./run_spatial_word_pca.sh "/path/to/model1|alias1" "/path/to/model2|alias2"
#
# 默认行为:
#   - 仅分析最后一层 (--layer -1) 并生成 PCA 3D 散点图
#   - 不弹出图形窗口 (--no_show)
#   - 如需分析所有层，在 python 调用中添加 --all_layers
#   - 输出到 output/pca_analysis/<model_alias>/ 目录
# ============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# 配置区 —— 按需修改
# ──────────────────────────────────────────────────────────────────

# 标注语料 JSON 文件路径
DATA_FILE="spatial_info_annotation/spatial_dataset_deepseek-v4-pro_20260618-143434_modified.json"

# 输出根目录（每个模型会在此下创建子目录）
OUTPUT_BASE_DIR="output/pca_analysis"

# Python 脚本路径
PYTHON_SCRIPT="spatial_word_pca.py"

# 模型列表：每项格式为 "模型路径|模型别名"
# 别名用于文件命名和输出子目录，建议使用英文/数字/连字符
MODELS=(
    # 示例（取消注释并修改为实际路径）:
    # "/data/models/Qwen2.5-7B-Instruct|qwen2.5-7b"
    # "/data/models/Llama-3-8B-Instruct|llama3-8b"
    # "/data/models/DeepSeek-R1-Distill-Qwen-7B|deepseek-r1-qwen7b"
)

LAYER=-1
# ──────────────────────────────────────────────────────────────────
# 解析命令行参数
# ──────────────────────────────────────────────────────────────────

if [[ $# -ge 1 ]]; then
    # 命令行传入的模型覆盖默认 MODELS 数组
    MODELS=("$@")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "============================================"
    echo "  错误: 未配置任何模型！"
    echo "============================================"
    echo ""
    echo "请通过以下任一方式指定模型："
    echo ""
    echo "  方式 1: 编辑脚本中的 MODELS 数组"
    echo "    格式: MODELS=("
    echo "        \"/path/to/model1|alias1\""
    echo "        \"/path/to/model2|alias2\""
    echo "    )"
    echo ""
    echo "  方式 2: 命令行传参"
    echo "    ./run_spatial_word_pca.sh \\"
    echo "        \"/path/to/model1|alias1\" \\"
    echo "        \"/path/to/model2|alias2\""
    echo ""
    exit 1
fi

# ──────────────────────────────────────────────────────────────────
# 环境检查
# ──────────────────────────────────────────────────────────────────

if [[ ! -f "$DATA_FILE" ]]; then
    echo "[错误] 数据文件不存在: $DATA_FILE"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "[错误] Python 脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 检查 Python 是否可用
if ! command -v python &>/dev/null; then
    echo "[错误] 未找到 Python，请确认已激活正确的 conda/venv 环境"
    exit 1
fi

echo "============================================"
echo "  空间词嵌入 PCA 批处理"
echo "============================================"
echo "  数据文件:   $DATA_FILE"
echo "  Python:     $(which python)"
echo "  模型数量:   ${#MODELS[@]}"
echo "  输出根目录: $OUTPUT_BASE_DIR"
echo "============================================"
echo ""

# ──────────────────────────────────────────────────────────────────
# 逐一处理每个模型
# ──────────────────────────────────────────────────────────────────

TOTAL=${#MODELS[@]}
CURRENT=0
FAILED_MODELS=()

for entry in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    # 解析 "路径|别名"
    IFS='|' read -r model_path model_alias <<< "$entry"

    if [[ -z "$model_path" || -z "$model_alias" ]]; then
        echo "[错误] 格式错误，需要 '模型路径|模型别名'，实际: '$entry'"
        FAILED_MODELS+=("$entry (格式错误)")
        continue
    fi

    if [[ ! -d "$model_path" ]]; then
        echo "[警告] 模型路径不存在，跳过: $model_path"
        FAILED_MODELS+=("$model_alias (路径不存在: $model_path)")
        continue
    fi

    output_dir="${OUTPUT_BASE_DIR}/${model_alias}"

    echo "============================================"
    echo "  [$CURRENT/$TOTAL] 模型: $model_alias"
    echo "  路径:   $model_path"
    echo "  输出:   $output_dir"
    echo "============================================"

    set +e  # 允许单个模型失败而不中断整个批处理
    python "$PYTHON_SCRIPT" \
        --data_file "$DATA_FILE" \
        --model_name_or_path "$model_path" \
        --model_alias "$model_alias" \
        --output_dir "$output_dir" \
        --no_show \
        --layer "$LAYER"
    exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
        echo ""
        echo "  ✓ 完成: $model_alias"
        # 列出生成的文件
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        echo "  生成 PNG: ${png_count// /} 张"
    else
        echo ""
        echo "  ✗ 失败: $model_alias (退出码: $exit_code)"
        FAILED_MODELS+=("$model_alias (退出码: $exit_code)")
    fi

    echo ""
done

# ──────────────────────────────────────────────────────────────────
# 汇总报告
# ──────────────────────────────────────────────────────────────────

echo "============================================"
echo "  批处理汇总"
echo "============================================"
echo "  总数:   $TOTAL"
echo "  成功:   $((TOTAL - ${#FAILED_MODELS[@]}))"
echo "  失败:   ${#FAILED_MODELS[@]}"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "  失败列表:"
    for item in "${FAILED_MODELS[@]}"; do
        echo "    - $item"
    done
fi

echo ""
echo "  输出目录: $OUTPUT_BASE_DIR"
echo "============================================"
