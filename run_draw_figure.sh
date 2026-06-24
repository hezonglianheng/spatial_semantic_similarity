#!/usr/bin/env bash
# =============================================================================
# run_draw_figure.sh
# =============================================================================
#
# 用于执行 draw_figure.py 的 bash 脚本。
# 你可以在本脚本的 "用户配置区" 中修改所有参数，然后直接运行:
#
#     bash run_draw_figure.sh
#
# 也可以从命令行传入参数覆盖默认值（见脚本末尾）。
#
# 依赖:
#   - Python 3.8+
#   - 需要安装的 Python 包: pandas, numpy, matplotlib
#     安装方式（任选一种）:
#       pip install pandas numpy matplotlib
#       conda install pandas numpy matplotlib
#
# =============================================================================

set -euo pipefail  # 遇到错误立即退出，未定义变量报错，管道中任一步失败都算失败

# =============================================================================
# 0. Python 环境配置
# =============================================================================

# ---- 如果你的环境需要激活 conda / venv，请取消下面相应行的注释 ----

# 方式 A: 使用 conda 环境
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate your_env_name

# 方式 B: 使用 Python venv（Windows Git Bash 下路径类似）
# source .venv/Scripts/activate

# 方式 C: 使用系统默认 python3，无需配置

# ---- Python 解释器选择 ----
# 如果 python 不在 PATH，可在此指定完整路径
PYTHON="${PYTHON:-python}"

# =============================================================================
# 1. 用户配置区 —— 在此修改所有参数
# =============================================================================

# ---------------------------------------------------------------------------
# 1.1 输入文件
# ---------------------------------------------------------------------------

# CSV 文件列表（必填）。支持通配符，如 "results_*.csv"
CSV_FILES=(
    "results.csv"
)
# 示例多文件:
# CSV_FILES=("results_qwen.csv" "results_llama.csv" "results_*.csv")

# 是否合并多个 CSV 文件（true = 合并为一个 DataFrame；false = 分别出图）
CONCAT=true

# ---------------------------------------------------------------------------
# 1.2 数据筛选
# ---------------------------------------------------------------------------

# 筛选条件数组（可选，留空表示不筛选）。
# 格式: "<列名> <操作符> <值>"
# 多个条件以 AND 逻辑组合。
#
# 支持的操作符一览:
#   == 或 =        等于
#   !=             不等于
#   >              大于
#   >=             大于等于
#   <              小于
#   <=             小于等于
#   between 或 []  区间，值用逗号分隔，如 "layer between 0,12"
#   in             属于集合，值用逗号分隔，如 "strategy in mean,last_token"
#   contains 或 has 字符串包含，如 "name contains test"
#   startswith     字符串前缀
#   endswith       字符串后缀
#
# 示例:
FILTERS=(
    # "accuracy > 0.5"
    # "strategy == mean"
    # "layer between 0,24"
    # "model_name in qwen,llama"
)

# ---------------------------------------------------------------------------
# 1.3 绘图核心参数
# ---------------------------------------------------------------------------

# X 轴列名（必填）
X_COLUMN="layer"

# Y 轴列名（必填，可多个，空格分隔表示多条折线）
Y_COLUMNS="accuracy"
# 示例多列:
# Y_COLUMNS="accuracy spearman_corr pearson_corr"

# 分组列（可选）。按此列的不同值分组，每组画一条折线（不同颜色）
# 多文件合并时使用 _source_file 可按文件区分颜色
GROUP_BY=""
# 示例: GROUP_BY="strategy"   GROUP_BY="_source_file"

# ---------------------------------------------------------------------------
# 1.4 输出设置
# ---------------------------------------------------------------------------

# 输出图片路径（可选）。支持 .png / .pdf / .svg。留空则尝试弹窗显示。
OUTPUT="output_figure.png"

# 图表标题（可选，留空则自动生成）
TITLE=""
# 示例: TITLE="Accuracy vs Layer"

# X 轴标签（可选，留空则使用列名）
X_LABEL=""

# Y 轴标签（可选，多个用空格分隔，与 Y_COLUMNS 一一对应；留空则使用列名）
Y_LABELS=""
# 示例: Y_LABELS="准确率 Spearman相关系数"

# ---------------------------------------------------------------------------
# 1.5 样式参数
# ---------------------------------------------------------------------------

# 绘图样式预设: default | ggplot | seaborn | fivethirtyeight
STYLE="default"

# 图像大小（宽 高，英寸）
FIG_WIDTH=10
FIG_HEIGHT=6

# 图像 DPI
DPI=150

# 折线图标记样式: o | s | ^ | D | . | "" 等（matplotlib marker 语法）
MARKER="o"

# 标记大小
MARKER_SIZE=5

# 线宽
LINE_WIDTH=1.8

# 图例位置: best | upper right | lower left | center | ...
LEGEND_LOC="best"

# 是否将图例放置在绘图区域外侧（true = 外侧，false = 内侧）
LEGEND_OUTSIDE=false

# 图例外侧锚点坐标（仅 LEGEND_OUTSIDE=true 时生效），格式: "x y"
LEGEND_BBOX="1.02 1.0"

# 是否显示网格（true = 显示，false = 隐藏）
SHOW_GRID=true

# 多条 Y 列时是否使用独立子图（true = 独立子图，false = 叠加在同一图）
SUBPLOTS=false

# ---------------------------------------------------------------------------
# 1.6 坐标轴边距
# ---------------------------------------------------------------------------

# X 轴范围边距比例（0.05 = 数据范围的 ±5%）
X_PADDING=0.05

# Y 轴范围边距比例（0.10 = 数据范围的 ±10%）
Y_PADDING=0.10

# ---------------------------------------------------------------------------
# 1.7 数据输出（可选）
# ---------------------------------------------------------------------------

# 是否在终端打印筛选后的数据
PRINT_FILTERED=false

# 将筛选后的数据保存为 CSV 文件（可选，留空则不保存）
SAVE_FILTERED=""
# 示例: SAVE_FILTERED="filtered_data.csv"

# =============================================================================
# 2. 脚本执行逻辑 —— 通常不需要修改以下内容
# =============================================================================

echo "============================================"
echo "  draw_figure.py 执行脚本"
echo "============================================"
echo ""

# 检查 Python 是否可用
if ! command -v "$PYTHON" &> /dev/null; then
    echo "[错误] 找不到 Python 解释器: $PYTHON"
    echo "       请修改 PYTHON 变量指向正确的 Python 路径，或激活 conda/venv 环境。"
    exit 1
fi
echo "[环境] Python: $("$PYTHON" --version 2>&1)"

# 检查 draw_figure.py 是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAW_FIGURE="$SCRIPT_DIR/draw_figure.py"

if [ ! -f "$DRAW_FIGURE" ]; then
    echo "[错误] 找不到 draw_figure.py: $DRAW_FIGURE"
    exit 1
fi
echo "[环境] 脚本路径: $DRAW_FIGURE"

# 构建命令
CMD=("$PYTHON" "$DRAW_FIGURE")

# ---- 文件参数 ----
CMD+=("--csv_files" "${CSV_FILES[@]}")
if [ "$CONCAT" = false ]; then
    CMD+=("--no_concat")
fi

# ---- 筛选参数 ----
if [ ${#FILTERS[@]} -gt 0 ]; then
    CMD+=("--filters" "${FILTERS[@]}")
fi

# ---- 绘图核心参数 ----
CMD+=("--x_column" "$X_COLUMN")
# 将空格分隔的 Y_COLUMNS 转为数组
IFS=' ' read -ra y_cols <<< "$Y_COLUMNS"
CMD+=("--y_columns" "${y_cols[@]}")

if [ -n "$GROUP_BY" ]; then
    CMD+=("--group_by" "$GROUP_BY")
fi

# ---- 输出参数 ----
if [ -n "$OUTPUT" ]; then
    CMD+=("--output" "$OUTPUT")
fi
if [ -n "$TITLE" ]; then
    CMD+=("--title" "$TITLE")
fi
if [ -n "$X_LABEL" ]; then
    CMD+=("--x_label" "$X_LABEL")
fi
if [ -n "$Y_LABELS" ]; then
    IFS=' ' read -ra y_lbls <<< "$Y_LABELS"
    CMD+=("--y_labels" "${y_lbls[@]}")
fi

# ---- 样式参数 ----
CMD+=("--style" "$STYLE")
CMD+=("--figsize" "$FIG_WIDTH" "$FIG_HEIGHT")
CMD+=("--dpi" "$DPI")
CMD+=("--marker" "$MARKER")
CMD+=("--markersize" "$MARKER_SIZE")
CMD+=("--linewidth" "$LINE_WIDTH")
CMD+=("--legend_loc" "$LEGEND_LOC")

if [ "$LEGEND_OUTSIDE" = true ]; then
    CMD+=("--legend_outside")
    # 解析 LEGEND_BBOX 为两个参数
    if [ -n "$LEGEND_BBOX" ]; then
        IFS=' ' read -ra bbox_parts <<< "$LEGEND_BBOX"
        CMD+=("--legend_bbox" "${bbox_parts[@]}")
    fi
fi

if [ "$SHOW_GRID" = false ]; then
    CMD+=("--no_grid")
fi
if [ "$SUBPLOTS" = true ]; then
    CMD+=("--subplots")
fi

# ---- 坐标轴边距 ----
CMD+=("--x_padding" "$X_PADDING")
CMD+=("--y_padding" "$Y_PADDING")

# ---- 数据输出 ----
if [ "$PRINT_FILTERED" = true ]; then
    CMD+=("--print_filtered")
fi
if [ -n "$SAVE_FILTERED" ]; then
    CMD+=("--save_filtered" "$SAVE_FILTERED")
fi

# ---- 如果从命令行传入了额外参数，追加到命令末尾 ----
if [ $# -gt 0 ]; then
    echo "[命令行] 追加额外参数: $*"
    CMD+=("$@")
fi

# ---- 执行 ----
echo ""
echo "[执行] 完整命令:"
printf "  %q " "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================"
    echo "  执行完成"
    echo "============================================"
else
    echo "============================================"
    echo "  执行失败（退出码: $EXIT_CODE）"
    echo "============================================"
fi
exit $EXIT_CODE
