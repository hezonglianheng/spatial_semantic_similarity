# encoding: utf-8
"""从空间词对中提取词嵌入，进行 PCA 降维与可视化。

采用与 spatial_word_embedding.py 相同的词嵌入提取方法：
将空间词填入 WORD_EMBEDDING_PROMPT 模板后编码，取后半部分 token 隐状态的
均值作为该空间词的词嵌入。

功能流程：
1. 读取标注语料 JSON 文件
2. 筛选 relation == "空间图式交集" 的记录
3. 从 pair 字段解析前后两个空间词（如 "上边-后边" → "上边"、"后边"）
4. 对每个空间词用 prompt 模板编码，取后半部分隐状态均值作为词嵌入（缓存去重）
5. 将空间词映射到基础空间词（上、下、前、后、里、外、旁）
6. 收集各基础空间词对应的全部词嵌入
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

import matplotlib.font_manager as fm

FONT_TTF_PATH = "/root/autodl-fs/simhei.ttf"
fm.fontManager.addfont(FONT_TTF_PATH)
_font_prop = fm.FontProperties(fname=FONT_TTF_PATH)
_font_name = _font_prop.get_name()
# 关键：必须把字体名加入到 sans-serif 列表的最前面，而不是直接设置 font.family
# 直接设 font.family = "SimHei" 会导致 matplotlib 找不到对应族而回退到 DejaVu Sans
plt.rcParams["font.sans-serif"] = [_font_name] + plt.rcParams["font.sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
print(f"[字体] 加载 TTF: {FONT_TTF_PATH} → {_font_name}")


# ══════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════

# 与 spatial_word_embedding.py 一致的 prompt 模板
WORD_EMBEDDING_PROMPT = r'被重复句:"{word}";重复句:"{word}"'

# 七类目标基础空间词（按词频排序，用于图例）
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


def get_all_layer_embeddings(
    word: str,
    extractor: extract_hidden_info.HiddenInfoExtractor,
) -> dict[int, list[float]]:
    """一次前向传播，返回一个空间词在所有层的嵌入向量。

    使用与 spatial_word_embedding.py 完全相同的方法：
    1. 将词填入 WORD_EMBEDDING_PROMPT 模板
    2. 计算前半部分（前缀）的 token 长度，确定后半部分起始位置
    3. 编码完整 prompt 获取所有层的 hidden_states
    4. 对每一层，提取后半部分 token 隐状态并取均值

    Args:
        word:      待编码的空间词。
        extractor: HiddenInfoExtractor 实例。

    Returns:
        {layer_index: embedding_list}，key 为 0-based 层索引。
    """
    prompt = WORD_EMBEDDING_PROMPT.format(word=word)

    # ── 计算前缀的 token 长度 ──
    # prompt 格式: 被重复句:"{word}";重复句:"{word}"
    # 前缀: 被重复句:"{word}";重复句:"
    prefix = f'被重复句:"{word}";重复句:"'
    prefix_tokens = extractor.tokenizer(
        [prefix],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    split = int(prefix_tokens.attention_mask.sum().item())

    # ── 编码完整 prompt ──
    batch_tokens = extractor.tokenizer(
        [prompt],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    total_len = int(batch_tokens.attention_mask.sum().item())

    with torch.no_grad():
        hs = extractor.encode(batch_tokens).hidden_states  # tuple of (1, L, D)

    # ── 逐层提取 ──
    result: dict[int, list[float]] = {}
    for layer_idx, layer_hs_tuple in enumerate(hs):
        layer_hs = layer_hs_tuple  # (1, L, D)  or  (1, L, D) depending on version
        word_tokens = layer_hs[0, split:total_len, :]      # (n_tokens, D)
        word_mean = word_tokens.mean(dim=0).cpu().tolist()
        result[layer_idx] = word_mean

    # 显式释放 GPU 显存
    del hs, batch_tokens, prefix_tokens

    return result


def get_single_layer_embedding(
    word: str,
    extractor: extract_hidden_info.HiddenInfoExtractor,
    layer: int,
) -> list[float]:
    """获取一个空间词在指定层的嵌入向量（兼容旧接口）。

    内部调用 get_all_layer_embeddings 并仅返回指定层的结果。
    """
    all_embs = get_all_layer_embeddings(word, extractor)
    num_layers = len(all_embs)
    if layer < 0:
        layer = num_layers + layer
    if layer not in all_embs:
        raise ValueError(f"层索引 {layer} 超出范围 [0, {num_layers - 1}]")
    return all_embs[layer]


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
    # Step 3: 收集所有不同的空间词
    # ──────────────────────────────────────────────────────────────
    all_words: set[str] = set()
    for rec in selected:
        pair = rec.pair
        if "-" in pair:
            parts = pair.split("-", 1)
            all_words.add(parts[0])
            all_words.add(parts[1])
        elif pair:
            all_words.add(pair)

    print(f"[信息] 不同空间词总数: {len(all_words)}")

    # 过滤出能映射到目标基础类的词
    encodable_words = sorted(w for w in all_words if extract_base(w) is not None)
    skipped_words = sorted(w for w in all_words if extract_base(w) is None)
    if skipped_words:
        print(f"[信息] 跳过非目标空间词 ({len(skipped_words)} 个): "
              f"{', '.join(skipped_words)}")

    print(f"[信息] 需编码的空间词: {len(encodable_words)} 个")

    # ──────────────────────────────────────────────────────────────
    # Step 4: 编码每个空间词（一次前向传播获取所有层）
    # ──────────────────────────────────────────────────────────────
    # 缓存结构: {word: {layer_idx: embedding_list}}
    word_all_layer_cache: dict[str, dict[int, list[float]]] = {}

    print(f"\n[编码] 共需编码 {len(encodable_words)} 个空间词（每词一次前向传播）：")
    for i, word in enumerate(encodable_words, 1):
        base = extract_base(word)
        try:
            all_layer_embs = get_all_layer_embeddings(word, extractor)
        except Exception:
            print(f"  [错误] 编码失败: {word}\n{traceback.format_exc()}")
            continue
        word_all_layer_cache[word] = all_layer_embs
        print(f"  ({i:>3}/{len(encodable_words)}) {word:　<6} → {base}  "
              f"[{len(all_layer_embs)} 层]")

    if not word_all_layer_cache:
        print("[错误] 没有成功编码任何空间词，退出。")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────
    # Step 5: 识别原始空间词与基础空间词的对应关系（全局共享）
    # ──────────────────────────────────────────────────────────────
    # 收集每个基础空间词下包含的原始空间词种类（用于打印信息）
    base_word_labels: dict[str, list[str]] = defaultdict(list)

    for rec in selected:
        pair = rec.pair
        if "-" not in pair:
            continue
        parts = pair.split("-", 1)
        for word in parts:
            if word not in word_all_layer_cache:
                continue
            base = extract_base(word)
            if base is None:
                continue
            base_word_labels[base].append(word)

    # ── 漏斗诊断：逐层统计数据损失 ──
    total_selected = len(selected)
    records_with_dash = 0
    records_without_dash = 0
    total_word_instances = 0       # 所有 pair 拆分后的词实例总数
    lost_not_encodable = 0         # 损失：词不含基础空间字符（从未编码）
    lost_encode_failed = 0         # 损失：词编码失败（不在 cache 中）
    lost_no_base = 0               # 损失：词无法映射到基础空间词
    kept_instances = 0             # 最终保留 = PCA 图中的点数

    for rec in selected:
        pair = rec.pair
        if "-" not in pair:
            records_without_dash += 1
            continue
        records_with_dash += 1
        parts = pair.split("-", 1)
        for word in parts:
            total_word_instances += 1
            # 检查这个词是否属于不可编码的类别（从未尝试编码）
            if extract_base(word) is None:
                lost_not_encodable += 1
                continue
            if word not in word_all_layer_cache:
                lost_encode_failed += 1
                continue
            # 二次 base 检查（理论上与上面一致，保留为安全网）
            if extract_base(word) is None:
                lost_no_base += 1
                continue
            kept_instances += 1

    print(f"\n[诊断] ═══════════════════ 数据漏斗 ═══════════════════")
    print(f"  筛选记录总数:           {total_selected:>6} 条")
    print(f"  含 '-' 的记录:          {records_with_dash:>6} 条  "
          f"(每记录拆为 2 词 → 预期 {records_with_dash * 2} 个词实例)")
    if records_without_dash:
        print(f"  不含 '-' 的记录:        {records_without_dash:>6} 条  (已丢弃，不参与 PCA)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  实际词实例总数:         {total_word_instances:>6} 个")
    print(f"  损失-不含基础字符:      {lost_not_encodable:>6} 个  (词中无 "
          f"{'/'.join(TARGET_BASES)} 任一字符)")
    print(f"  损失-编码失败:          {lost_encode_failed:>6} 个  (不在编码缓存中)")
    if lost_no_base:
        print(f"  损失-无法映射基础词:    {lost_no_base:>6} 个")
    print(f"  ─────────────────────────────────────────────")
    print(f"  ★ 最终 PCA 点数:        {kept_instances:>6} 个")
    print(f"[诊断] ═══════════════════════════════════════════")

    # 打印各基础空间词的原始词分布
    print(f"\n[统计] 各基础空间词包含的原始空间词：")
    for base in TARGET_BASES:
        wl = base_word_labels.get(base, [])
        if wl:
            unique_words = sorted(set(wl))
            print(f"  {base}: {len(wl)} 个实例 → {', '.join(unique_words)}")
        else:
            print(f"  {base}: 0 个")

    # ──────────────────────────────────────────────────────────────
    # Step 6: 保存全层嵌入数据（JSON）
    # ──────────────────────────────────────────────────────────────
    # 组织为 {layer_idx: {base: [embeddings]}}
    layer_embeddings: dict[int, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in selected:
        pair = rec.pair
        if "-" not in pair:
            continue
        parts = pair.split("-", 1)
        for word in parts:
            if word not in word_all_layer_cache:
                continue
            base = extract_base(word)
            if base is None:
                continue
            for layer_idx, emb in word_all_layer_cache[word].items():
                layer_embeddings[layer_idx][base].append(emb)

    '''
    # 序列化为 JSON 兼容结构
    embed_save_path = os.path.join(
        args.output_dir,
        f"spatial_word_embeddings_{args.model_alias}.json",
    )
    save_data: dict = {
        "model_alias": args.model_alias,
        "model_type": model_info["model_type"],
        "mode": mode,
        "num_layers": num_layers,
        "hidden_size": model_info["hidden_size"],
        "target_bases": TARGET_BASES,
        "base_word_labels": {
            base: sorted(set(base_word_labels.get(base, [])))
            for base in TARGET_BASES
        },
        "layer_embeddings": {
            str(li): {
                base: embs for base, embs in base_dict.items()
            }
            for li, base_dict in sorted(layer_embeddings.items())
        },
    }
    with open(embed_save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n[信息] 已保存词嵌入数据到 {embed_save_path}")
    '''

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
