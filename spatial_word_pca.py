# encoding: utf-8
"""从空间词对中提取上下文相关词嵌入，进行 PCA 降维与可视化。

采用上下文相关编码方法（方案 A）：
通过字符级差异定位每个空间词在完整句子中的位置，编码整句后提取该词所在
token 的隐状态均值作为词嵌入。同一空间词（如"上面"）在不同句子中会产生
不同的嵌入向量，反映上下文语义差异。

功能流程：
1. 读取标注语料 JSON 文件
2. 筛选 relation == "空间图式交集" 的记录
3. 对每条记录，通过字符差异定位两个空间词各自的字符区间
4. 编码完整句子，提取空间词对应 token 在各层的隐状态均值（不缓存去重）
5. 将空间词映射到基础空间词（上、下、前、后、里、外、旁）
6. 收集各基础空间词对应的全部词嵌入（每个样本独立）
7. PCA 降维至 3 维，不同基础空间词用不同颜色绘制散点图

支持 --all_layers：一次前向传播提取所有层的嵌入，逐层 PCA 并保存图片。
"""

import extract_hidden_info
import read_data
from pca_analysis import pca_reduce

import argparse
import os
import json
import sys
import traceback
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")              # 非交互后端，批处理不需要显示器
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# 中文字体配置 —— 直接使用 TTF 字体文件
# ══════════════════════════════════════════════════════════════════════
#
# 为什么不能只设 font.family = "SimHei"？
# ─────────────────────────────────────
# matplotlib 的字体解析不是按字体名直接查找，而是通过"通用族"体系：
#   font.family = "sans-serif"
#       → 查 font.sans-serif = ["DejaVu Sans", "Arial", ...]  按序尝试
# 把 "SimHei" 直接赋给 font.family 会被当成一个不存在的通用族名，
# matplotlib 找不到后回退到默认的 sans-serif 族 → DejaVu Sans。
#
# 为什么 addfont() 后可能不生效？
# ─────────────────────────────────
# matplotlib 将字体列表缓存到磁盘（~/.cache/matplotlib/fontlist*.json）。
# 如果缓存在 addfont() 之前已经存在，matplotlib 在渲染时可能使用缓存的
# 字体列表，导致新添加的字体被忽略。必须先清除缓存再添加字体。

import matplotlib.font_manager as fm

FONT_TTF_PATH = "/root/autodl-fs/simhei.ttf"

# ---- Step 1: 清除 matplotlib 字体缓存 ----
_cache_cleared = False
for _cache_base in (
    matplotlib.get_cachedir(),
    os.path.join(os.path.expanduser("~"), ".matplotlib"),
    os.path.join(os.path.expanduser("~"), ".cache", "matplotlib"),
):
    if os.path.isdir(_cache_base):
        for _fname in os.listdir(_cache_base):
            if _fname.startswith("fontlist") or _fname.startswith("fontList"):
                _cache_path = os.path.join(_cache_base, _fname)
                try:
                    os.remove(_cache_path)
                    print(f"[字体] 删除字体缓存: {_cache_path}")
                    _cache_cleared = True
                except OSError:
                    pass

if _cache_cleared:
    # 强制重建 FontManager，重新扫描系统所有字体
    fm._load_fontmanager(try_read_cache=False)

# ---- Step 2: 加载字体文件 ----
if not os.path.exists(FONT_TTF_PATH):
    raise FileNotFoundError(
        f"字体文件不存在: {FONT_TTF_PATH}\n"
        f"请确认 simhei.ttf 文件路径是否正确，或修改 FONT_TTF_PATH 变量。"
    )

fm.fontManager.addfont(FONT_TTF_PATH)
_font_prop = fm.FontProperties(fname=FONT_TTF_PATH)
_font_name = _font_prop.get_name()

# ---- Step 3: 设置 matplotlib 全局字体 ----
plt.rcParams["font.sans-serif"] = [_font_name] + plt.rcParams["font.sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

print(f"[字体] 加载 TTF: {FONT_TTF_PATH} → {_font_name}")

# 验证字体已正确注册到 fontManager
_ttf_entries = [f for f in fm.fontManager.ttflist if f.name == _font_name]
print(f"[字体] fontManager 中匹配 '{_font_name}' 的条目: {len(_ttf_entries)} 个")


# ══════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════

# 七类目标基础空间词（按词频排序，用于图例）
# 句子重复 prompt：将句子重复两次，取第二次重复中差异部分的嵌入
# 与 spatial_word_embedding.py 保持一致
WORD_EMBEDDING_PROMPT = r'被重复句:"{sentence}";重复句:"{sentence}"'

TARGET_BASES = ["上", "下", "前", "后", "里", "外", "旁"]

# 为每类基础空间词分配固定颜色（兼顾色盲友好与显示区分度）
BASE_COLORS: dict[str, str] = {
    "上": "#E63946",   # 红
    "下": "#457B9D",   # 蓝
    "前": "#2A9D8F",   # 青绿
    "后": "#E9C46A",   # 金
    "里": "#F4A261",   # 橙
    "外": "#9B5DE5",   # 紫
    "旁": "#06D6A0",   # 薄荷绿
}


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def extract_base(word: str) -> str | None:
    """从复合空间词中提取基础空间词。

    匹配规则：只要复合词中包含某个基础空间字符，即认为属于该基础类别。
    例如 "上边"、"上面"、"之上" 均映射到 "上"。

    Args:
        word: 复合空间词，如 "上边"、"里面"、"旁边" 等。

    Returns:
        基础空间词（上/下/前/后/里/外/旁），无法匹配时返回 None。
    """
    for base in TARGET_BASES:
        if base in word:
            return base
    return None


def find_char_diff_span(s1: str, s2: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """找出两个仅相差一个空间词的句子的字符级差异区间。

    从两端向中间同时扫描，找出公共前缀与公共后缀，剩余部分即为差异区间。
    例如 s1="…在上边…"、s2="…在后边…" → 返回 ((i, j1), (i, j2))，
    分别代表 s1 中"上边"的区间和 s2 中"后边"的区间。

    Args:
        s1: 句子 1（含空间词 word1）
        s2: 句子 2（含空间词 word2）

    Returns:
        ((start1, end1), (start2, end2)) —— 各自句子中差异部分的 [start, end) 区间。
    """
    # 公共前缀
    i = 0
    while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
        i += 1

    # 公共后缀（从末尾向 i 收缩）
    end1, end2 = len(s1), len(s2)
    while end1 > i and end2 > i and s1[end1 - 1] == s2[end2 - 1]:
        end1 -= 1
        end2 -= 1

    return (i, end1), (i, end2)


def find_word_in_sentence(
    sentence: str,
    word: str,
    hint_pos: int | None = None,
) -> tuple[int, int] | None:
    """在句子中定位空间词的字符区间 [start, end)。

    优先使用 hint_pos（差异区间起点）来消除歧义：
    当空间词在句子中出现多次时，选取最靠近 hint_pos 的那一次。

    Args:
        sentence: 完整句子。
        word:     空间词（来自 pair 字段）。
        hint_pos: 差异区间的字符偏移（由 find_char_diff_span 提供）。

    Returns:
        (start, end) 或 None（未找到）。
    """
    # 收集所有出现位置
    occurrences: list[int] = []
    start = 0
    while True:
        pos = sentence.find(word, start)
        if pos == -1:
            break
        occurrences.append(pos)
        start = pos + 1

    if not occurrences:
        return None

    if len(occurrences) == 1:
        return occurrences[0], occurrences[0] + len(word)

    # 多个出现位置：选最接近 hint_pos 的
    if hint_pos is not None:
        best = min(occurrences, key=lambda p: abs(p - hint_pos))
    else:
        best = occurrences[0]

    return best, best + len(word)


def get_contextual_word_embedding(
    sentence: str,
    char_start: int,
    char_end: int,
    extractor: extract_hidden_info.HiddenInfoExtractor,
) -> dict[int, list[float]]:
    """用「句子重复两次」的方式编码，提取空间词在第二次重复中各层的隐状态均值。

    与 spatial_word_embedding.py 采用相同的编码策略：
    1. 对原始句子做 tokenize（带 offset_mapping），定位空间词对应的 token 索引
    2. 将句子填入 WORD_EMBEDDING_PROMPT → "被重复句:{sentence};重复句:{sentence}"
    3. 对 prompted 文本 tokenize，计算第二次重复的起始 token 位置（split）
    4. 将原始句子的 token 索引映射到 prompted 文本的第二半部分
    5. 一次前向传播获取所有层的 hidden_states
    6. 逐层提取第二半部分中目标 token 的隐状态并取算术均值

    这样做的动机：模型在第二次重复时已经"读过"整个句子，对空间词的编码
    更充分地融合了上下文语义，相比直接编码原始句子能获得更稳定的上下文表示。

    Args:
        sentence:   包含空间词的完整句子。
        char_start: 空间词在句子中的起始字符偏移。
        char_end:   空间词在句子中的结束字符偏移（不含）。
        extractor:  HiddenInfoExtractor 实例。

    Returns:
        {layer_index: embedding_list}，key 为 0-based 层索引。
    """
    # ── Step 1: 对原始句子 tokenize，定位空间词的 token 索引 ──
    raw_batch = extractor.tokenizer(
        [sentence],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
    )
    raw_offset_mapping = raw_batch.pop("offset_mapping")[0].tolist()

    # 筛选与 [char_start, char_end) 有重叠的 token 索引
    raw_token_indices: list[int] = []
    for idx, (t_cs, t_ce) in enumerate(raw_offset_mapping):
        if t_cs < char_end and t_ce > char_start:
            raw_token_indices.append(idx)

    if not raw_token_indices:
        raise ValueError(
            f"未找到字符区间 [{char_start}, {char_end}) 对应的 token。"
            f"句子前 60 字: {sentence[:60]!r}"
        )

    # ── Step 2: 构建 prompted 文本 ──
    prompt = WORD_EMBEDDING_PROMPT.format(sentence=sentence)

    # ── Step 3: 对 prompted 文本 tokenize ──
    batch_tokens = extractor.tokenizer(
        [prompt],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )

    # ── Step 4: 计算第二次重复的起始 token 位置 ──
    # prompt 格式: 被重复句:"{sentence}";重复句:"{sentence}"
    # 前半部分: 被重复句:"{sentence}";重复句:"
    # 后半部分: {sentence}"  ← 只取这部分向量
    prefix = f'被重复句:"{sentence}";重复句:"'
    prefix_tokens = extractor.tokenizer(
        [prefix],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    split = int(prefix_tokens.attention_mask.sum().item())

    # ── Step 5: 将原始句子 token 索引映射到 prompted 文本的第二半部分 ──
    prompted_token_indices = [i + split for i in raw_token_indices]

    # ── Step 6: 编码 prompted 文本 ──
    with torch.no_grad():
        hs = extractor.encode(batch_tokens).hidden_states  # tuple of (1, L, D)

    # ── Step 7: 逐层提取第二半部分中目标 token 的隐状态均值 ──
    result: dict[int, list[float]] = {}
    for layer_idx, layer_hs in enumerate(hs):
        word_tokens = layer_hs[0, prompted_token_indices, :]  # (n_tokens, D)
        word_mean = word_tokens.mean(dim=0).cpu().tolist()
        result[layer_idx] = word_mean

    # 显式释放 GPU 显存
    del hs, batch_tokens, raw_batch, prefix_tokens

    return result


# ══════════════════════════════════════════════════════════════════════
# PCA 可视化（单层）
# ══════════════════════════════════════════════════════════════════════

def plot_and_save_layer(
    vectors: np.ndarray,
    labels: np.ndarray,
    present_bases: list[str],
    layer_idx: int,
    model_alias: str,
    output_dir: str,
) -> str | None:
    """对某一层的词嵌入做 PCA 降维到 3D，绘制散点图并保存。

    Args:
        vectors:       shape (n_samples, hidden_dim)
        labels:        shape (n_samples,)，值为基础空间词字符串
        present_bases: 实际存在的基础空间词列表
        layer_idx:     当前层索引（用于标题和文件名）
        model_alias:   模型别名
        output_dir:    输出目录

    Returns:
        保存的图片路径，PCA 失败时返回 None。
    """
    n_samples = vectors.shape[0]
    if n_samples < 4:
        print(f"  [跳过] layer={layer_idx}: 样本数不足 ({n_samples} < 4)")
        return None

    try:
        reduced, pca_obj = pca_reduce(vectors, n_components=3, scale=True)
    except Exception:
        print(f"  [警告] layer={layer_idx}: PCA 失败\n{traceback.format_exc()}")
        return None

    evr = pca_obj.explained_variance_ratio_
    print(f"  layer={layer_idx:>3}: "
          f"PC1={evr[0]:.4f} PC2={evr[1]:.4f} PC3={evr[2]:.4f} "
          f"累计={evr.sum():.4f}")

    # ── 绘图 ──
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    for base in present_bases:
        mask = labels == base
        color = BASE_COLORS.get(base, "gray")
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            reduced[mask, 2],
            label=base,
            alpha=0.8,
            s=60,
            c=color,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(
        f"PCA 3D — 空间词嵌入\n"
        f"模型: {model_alias}  |  layer={layer_idx}  |  样本: {n_samples}"
    )
    ax.legend(loc="best", title="基础空间词")
    fig.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"spatial_word_pca_3d_{model_alias}_layer{layer_idx}.png",
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


# ══════════════════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════════════════

def main():
    argparser = argparse.ArgumentParser(
        description="提取空间词词嵌入 → PCA 3D 降维 → 可视化"
    )
    argparser.add_argument(
        "--data_file", required=True,
        help="标注语料 JSON 文件路径（如 spatial_dataset_xxx_modified.json）",
    )
    argparser.add_argument(
        "--model_name_or_path", required=True,
        help="模型路径或 HuggingFace 模型名称",
    )
    argparser.add_argument(
        "--model_alias", default="model",
        help="用于文件命名的模型别名（默认: model）",
    )
    argparser.add_argument(
        "--output_dir", default=".",
        help="输出目录（默认: 当前目录）",
    )
    argparser.add_argument(
        "--layer", type=int, default=None,
        help="仅分析指定层（默认: -1 即最后一层）。"
             "0=embedding 层，与 --all_layers 互斥",
    )
    argparser.add_argument(
        "--all_layers", action="store_true",
        help="对所有层逐一分析并生成各自的 PCA 图",
    )
    argparser.add_argument(
        "--no_show", action="store_true", default=True,
        help="不弹出图形窗口（批处理模式默认启用）",
    )
    argparser.add_argument(
        "--show", action="store_true",
        help="弹出图形窗口（覆盖 --no_show）",
    )

    args = argparser.parse_args()

    # ── 确定运行模式 ──
    if args.layer is not None and args.all_layers:
        print("[错误] --layer 与 --all_layers 互斥，请只指定其中一个。")
        sys.exit(1)

    if args.all_layers:
        mode = "all_layers"
    else:
        mode = "single"       # 默认仅分析最后一层
        if args.layer is None:
            args.layer = -1

    show = args.show and not args.no_show  # --show 覆盖 --no_show

    os.makedirs(args.output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Step 1: 加载模型
    # ──────────────────────────────────────────────────────────────
    print(f"[信息] 加载模型: {args.model_name_or_path}")
    extractor = extract_hidden_info.HiddenInfoExtractor(args.model_name_or_path)
    model_info = extractor.get_model_info()
    num_layers = model_info["num_hidden_states"]
    print(f"[信息] 模型类型: {model_info['model_type']}, "
          f"hidden_size={model_info['hidden_size']}, "
          f"num_hidden_states={num_layers}")

    if mode == "single":
        actual_layer = args.layer
        if actual_layer < 0:
            actual_layer = num_layers + actual_layer
        if actual_layer < 0 or actual_layer >= num_layers:
            print(f"[错误] layer={args.layer} 超出范围 "
                  f"[{-num_layers}, {num_layers - 1}]")
            sys.exit(1)
        target_layers = [actual_layer]
        print(f"[信息] 单层模式: 第 {actual_layer} 层")
    else:
        target_layers = list(range(num_layers))
        print(f"[信息] 全层模式: 共 {num_layers} 层 (0 ~ {num_layers - 1})")

    # ──────────────────────────────────────────────────────────────
    # Step 2: 加载数据集并筛选
    # ──────────────────────────────────────────────────────────────
    print(f"\n[信息] 加载数据: {args.data_file}")
    data = read_data.SpatialDataset(args.data_file)
    print(f"[信息] 数据集共 {len(data)} 条记录")

    selected = data.filter_by_relation("空间图式交集")
    print(f"[信息] 筛选 '空间图式交集': {len(selected)} 条")

    if not selected:
        print("[错误] 没有匹配的记录，退出。")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────
    # Step 3-5: 逐句编码 —— 每个空间词从其句子上下文中提取嵌入
    # ──────────────────────────────────────────────────────────────
    # 方案 A：不再用孤立 prompt 模板编码，而是编码完整句子，
    # 通过字符差异定位空间词，从句子隐状态中提取该词的上下文相关嵌入。
    # 同一"上面"在不同句子中会产生不同的嵌入向量，PCA 图中呈现为不同点。
    #
    # 数据结构：
    #   layer_embeddings: {layer_idx: {base: [embedding_list]}}
    #   base_word_labels: {base: [word_str]}  —— 用于打印分布统计

    layer_embeddings: dict[int, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    base_word_labels: dict[str, list[str]] = defaultdict(list)

    # ── 漏斗统计 ──
    total_selected = len(selected)
    records_with_dash = 0
    records_without_dash = 0
    total_word_instances = 0
    lost_no_base = 0           # 词不含基础空间字符（如上/下/前/后…）
    lost_not_found = 0         # 两句中都找不到 pair 字段声明的空间词
    lost_encode_failed = 0     # 模型编码失败
    kept_instances = 0         # 最终保留 = PCA 图中的点数

    print(f"\n[编码] 共 {len(selected)} 条记录，逐句编码空间词（每个词从其句子上下文中提取）：")

    for rec_idx, rec in enumerate(selected):
        pair = rec.pair
        if "-" not in pair:
            records_without_dash += 1
            continue
        records_with_dash += 1

        w1, w2 = pair.split("-", 1)
        s1, s2 = rec.sentence1, rec.sentence2

        # ── 定位差异区间 ──
        (cs1, ce1), (cs2, ce2) = find_char_diff_span(s1, s2)

        for word in [w1, w2]:
            total_word_instances += 1

            # 1) 基础空间词映射
            base = extract_base(word)
            if base is None:
                lost_no_base += 1
                continue

            # 2) 在两句中分别搜索该空间词（以各自差异区间为线索消除歧义）
            #    pair 字段的词序不一定与句子一致（如 pair="后-里" 但 s1 含"里"），
            #    因此对每句都尝试查找，优先选取离差异区间更近的出现。
            span1 = find_word_in_sentence(s1, word, hint_pos=cs1)
            span2 = find_word_in_sentence(s2, word, hint_pos=cs2)

            if span1 is None and span2 is None:
                lost_not_found += 1
                continue

            # 选择更靠近差异区间的那个句子（若只有一句找到则直接用那句）
            if span1 is not None and span2 is not None:
                # 判断哪个更接近自身的差异区间
                d1 = abs(span1[0] - cs1)
                d2 = abs(span2[0] - cs2)
                if d1 <= d2:
                    sentence, (cs, ce) = s1, span1
                else:
                    sentence, (cs, ce) = s2, span2
            elif span1 is not None:
                sentence, (cs, ce) = s1, span1
            else:
                sentence, (cs, ce) = s2, span2

            # 3) 编码句子并提取空间词的上下文嵌入
            try:
                all_layer_embs = get_contextual_word_embedding(
                    sentence, cs, ce, extractor,
                )
            except Exception:
                lost_encode_failed += 1
                if lost_encode_failed <= 5:       # 只打印前 5 个错误，避免刷屏
                    print(f"  [错误] 编码失败: word={word!r} "
                          f"span=({cs}, {ce}) sentence[:60]={sentence[:60]!r}")
                    traceback.print_exc()
                continue

            # 5) 按层归类
            for layer_idx, emb in all_layer_embs.items():
                layer_embeddings[layer_idx][base].append(emb)

            base_word_labels[base].append(word)
            kept_instances += 1

        # 进度
        if (rec_idx + 1) % 100 == 0:
            print(f"  进度: {rec_idx + 1}/{total_selected} 条记录, "
                  f"已保留 {kept_instances} 个词嵌入")

    print(f"  完成: {total_selected}/{total_selected} 条记录, "
          f"共保留 {kept_instances} 个上下文词嵌入")

    if kept_instances == 0:
        print("[错误] 没有成功提取任何上下文词嵌入，退出。")
        sys.exit(1)

    # ── 漏斗诊断 ──
    print(f"\n[诊断] ═══════════════════ 数据漏斗 ═══════════════════")
    print(f"  筛选记录总数:           {total_selected:>6} 条")
    print(f"  含 '-' 的记录:          {records_with_dash:>6} 条  "
          f"(每记录拆为 2 词 → 预期 {records_with_dash * 2} 个词实例)")
    if records_without_dash:
        print(f"  不含 '-' 的记录:        {records_without_dash:>6} 条  (已丢弃)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  实际词实例总数:         {total_word_instances:>6} 个")
    print(f"  损失-不含基础字符:      {lost_no_base:>6} 个  (词中无 "
          f"{'/'.join(TARGET_BASES)} 任一字符)")
    print(f"  损失-两句都找不到词:    {lost_not_found:>6} 个  (pair 字段词与句子内容不一致)")
    print(f"  损失-编码失败:          {lost_encode_failed:>6} 个")
    print(f"  ─────────────────────────────────────────────")
    print(f"  ★ 最终 PCA 点数:        {kept_instances:>6} 个  "
          f"(每个点=一个空间词在其句子上下文中的嵌入)")
    print(f"[诊断] ═══════════════════════════════════════════")

    # 打印各基础空间词的原始词分布
    print(f"\n[统计] 各基础空间词包含的原始空间词：")
    for base in TARGET_BASES:
        wl = base_word_labels.get(base, [])
        if wl:
            unique_words = sorted(set(wl))
            print(f"  {base}: {len(wl)} 个实例 ({len(unique_words)} 种) "
                  f"→ {', '.join(unique_words)}")
        else:
            print(f"  {base}: 0 个")

    # ──────────────────────────────────────────────────────────────
    # Step 7: 逐层 PCA + 绘图
    # ──────────────────────────────────────────────────────────────
    print(f"\n[PCA] 逐层降维与绘图 ({len(target_layers)} 层)...")

    # 统计所有层的 present_bases（用于保持图例一致）
    all_present_bases = [
        b for b in TARGET_BASES
        if any(len(layer_embeddings[li][b]) > 0 for li in target_layers)
    ]

    success_count = 0
    for layer_idx in target_layers:
        # 构建该层的向量矩阵和标签
        vectors_list: list[list[float]] = []
        labels_list: list[str] = []

        for base in TARGET_BASES:
            embs = layer_embeddings[layer_idx].get(base, [])
            for emb in embs:
                vectors_list.append(emb)
                labels_list.append(base)

        if not vectors_list:
            print(f"  [跳过] layer={layer_idx}: 无可用嵌入")
            continue

        vec = np.array(vectors_list, dtype=np.float64)
        lab = np.array(labels_list)

        saved = plot_and_save_layer(
            vec, lab, all_present_bases,
            layer_idx, args.model_alias, args.output_dir,
        )
        if saved:
            success_count += 1

    print(f"\n[信息] 成功生成 {success_count}/{len(target_layers)} 张 PCA 图")

    # ──────────────────────────────────────────────────────────────
    # Step 8: （可选）显示最后一张图
    # ──────────────────────────────────────────────────────────────
    if show and success_count > 0:
        # 重新绘制最后一层用于交互显示
        last_layer = target_layers[-1]
        vectors_list = []
        labels_list = []
        for base in TARGET_BASES:
            embs = layer_embeddings[last_layer].get(base, [])
            for emb in embs:
                vectors_list.append(emb)
                labels_list.append(base)

        if vectors_list:
            vec = np.array(vectors_list, dtype=np.float64)
            lab = np.array(labels_list)
            reduced, _ = pca_reduce(vec, n_components=3, scale=True)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection="3d")
            for base in all_present_bases:
                mask = lab == base
                ax.scatter(
                    reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                    label=base, alpha=0.8, s=60,
                    c=BASE_COLORS.get(base, "gray"),
                    edgecolors="white", linewidth=0.3,
                )
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.set_title(
                f"PCA 3D — 空间词嵌入\n"
                f"模型: {args.model_alias}  |  layer={last_layer}  |  "
                f"样本: {len(lab)}"
            )
            ax.legend(loc="best", title="基础空间词")
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    print("[完成]")


if __name__ == "__main__":
    main()
