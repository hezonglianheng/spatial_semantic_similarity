# encoding: utf-8
"""通过空间差异区域的词向量计算句子相似度。

与 simple_similarity.py 不同，本脚本不直接比较整个句子的句向量，而是：
1. 找出两个句子分词后的差异区域（diff regions）
2. 提取差异区域对应的 token 级隐状态
3. 对差异区域的 token 向量取均值
4. 计算两个句子差异区域均值向量的余弦相似度

这样可以更精准地捕获空间语义差异带来的影响。
"""

import extract_hidden_info
import read_data

from difflib import SequenceMatcher
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F


def get_diff(seq1: list[int], seq2: list[int]) -> list[tuple[slice, slice]]:
    """找出两个分词序列之间的差异区域。

    使用 SequenceMatcher 比对两个 token id 序列，返回每一处差异
    在两个序列中各自对应的切片，后续可直接用于提取差异部分的词向量。

    Args:
        seq1: 第一个分词序列（token id 列表）。
        seq2: 第二个分词序列（token id 列表）。

    Returns:
        (slice1, slice2) 列表，每个元素对应该处差异在 seq1/seq2 中的切片。
        完全一致时返回空列表。
    """
    diff_regions: list[tuple[slice, slice]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, seq1, seq2).get_opcodes():
        if tag == "equal":
            continue
        diff_regions.append((slice(i1, i2), slice(j1, j2)))
    return diff_regions


def collect_diff_indices(
    diff_regions: list[tuple[slice, slice]], side: int
) -> list[int]:
    """从差异区域列表中收集指定侧的 token 索引（去重排序）。

    Args:
        diff_regions: get_diff 返回的差异区域列表。
        side: 0 表示第一句（使用 slice1），1 表示第二句（使用 slice2）。

    Returns:
        去重排序后的索引列表。若某侧无差异 token（纯插入/删除场景）则返回空列表。
    """
    indices: list[int] = []
    for sl1, sl2 in diff_regions:
        sl = sl1 if side == 0 else sl2
        indices.extend(range(sl.start, sl.stop))
    return sorted(set(indices))


def build_diff_vector_record(
    idx: int,
    record: read_data.SentencePairRecord,
    s1_ids: list[int],
    s2_ids: list[int],
    s1_tokens: list[str],
    s2_tokens: list[str],
    diff_regions: list[tuple[slice, slice]],
    hidden_states: tuple[torch.Tensor, ...] | None,
    num_layers: int,
) -> dict:
    """为单个样本构建包含原始数据与差异向量的字典记录。

    Args:
        idx: 样本序号。
        record: 原始数据记录（SpatialDataset 中的一条）。
        s1_ids / s2_ids: 两句的 token id 序列（不含 padding）。
        s1_tokens / s2_tokens: 两句的 token 字符串列表。
        diff_regions: get_diff 返回的差异区域列表。
        hidden_states: 模型编码输出的 hidden_states 元组，每层 shape (2, L_max, D)。
        num_layers: 总层数（含 embedding 层）。

    Returns:
        字典，包含原始数据字段、差异区域信息、以及各层差异均值向量。
    """
    record_dict: dict = {
        "idx": idx,
        "id": record.id,
        "sentence1": record.sentence1,
        "sentence2": record.sentence2,
        "label": record.label,
        "pair": record.pair,
        "relation": record.relation,
        "s1_tokens": s1_tokens,
        "s2_tokens": s2_tokens,
        "diff_regions": [
            {
                "s1_slice": (sl1.start, sl1.stop),
                "s2_slice": (sl2.start, sl2.stop),
                "s1_tokens": s1_tokens[sl1.start:sl1.stop],
                "s2_tokens": s2_tokens[sl2.start:sl2.stop],
            }
            for sl1, sl2 in diff_regions
        ],
    }

    # 附加标注信息（若存在）
    if record.has_annotation:
        record_dict["target1"] = record.target1
        record_dict["reference1"] = record.reference1
        record_dict["target2"] = record.target2
        record_dict["reference2"] = record.reference2

    if hidden_states is None or not diff_regions:
        # 无编码结果或两句完全一致 → 所有层向量置为 None
        for layer in range(num_layers):
            record_dict[f"layer_{layer}_s1_diff_mean"] = None
            record_dict[f"layer_{layer}_s2_diff_mean"] = None
        return record_dict

    # 收集差异区域索引
    s1_diff_indices = collect_diff_indices(diff_regions, side=0)
    s2_diff_indices = collect_diff_indices(diff_regions, side=1)

    for layer in range(num_layers):
        layer_hs = hidden_states[layer]  # (2, L_max, D)
        if s1_diff_indices:
            s1_diff = layer_hs[0, s1_diff_indices, :]  # (n1_diff, D)
            s1_mean = s1_diff.mean(dim=0).cpu().numpy()
        else:
            s1_mean = None

        if s2_diff_indices:
            s2_diff = layer_hs[1, s2_diff_indices, :]  # (n2_diff, D)
            s2_mean = s2_diff.mean(dim=0).cpu().numpy()
        else:
            s2_mean = None

        record_dict[f"layer_{layer}_s1_diff_mean"] = s1_mean
        record_dict[f"layer_{layer}_s2_diff_mean"] = s2_mean

    return record_dict


def main():
    argparser = argparse.ArgumentParser(description="通过空间差异区域的词向量计算句子相似度")
    argparser.add_argument("--data_file", help="数据文件路径")
    argparser.add_argument("--model_name_or_path", help="模型路径或名称")
    argparser.add_argument("--model_alias", help="用于记录的模型名称")
    argparser.add_argument("--output_dir", default=".", help="输出目录，默认为当前目录")
    argparser.add_argument(
        "--save_diff_vectors",
        action="store_true",
        default=False,
        help="保存每个样本的差异区域向量及原始数据到 .pkl 文件",
    )

    args = argparser.parse_args()

    # 初始化隐藏信息提取器
    extractor = extract_hidden_info.HiddenInfoExtractor(args.model_name_or_path)
    # 读取数据
    data = read_data.SpatialDataset(args.data_file)
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)

    labels = np.array(data.labels)

    # 先用第一条数据探测 hidden_states 的实际层数（含 embedding 层）
    first_s1, first_s2 = data.sentences[0]
    with torch.no_grad():
        first_tokens = extractor.tokenizer(
            [first_s1], add_special_tokens=False, return_tensors="pt"
        )
        first_hs = extractor.encode(first_tokens).hidden_states
    num_layers = len(first_hs)

    cosine_sim_matrix = np.zeros((len(data.sentences), num_layers))

    # 用于保存差异向量数据集（仅当 --save_diff_vectors 开启时收集）
    diff_vector_dataset: list[dict] = []

    # 进度条（无 tqdm 时静默回退）
    try:
        from tqdm import tqdm
        iterator = tqdm(
            enumerate(data), total=len(data),
            desc="[空间相似度]"
        )
    except ImportError:
        iterator = enumerate(data)

    for idx, record in iterator:
        sentence1, sentence2 = record.sentence1, record.sentence2

        # ── 批量分词：两句一起 tokenize，减少后续 encode 调用次数 ──
        batch_tokens = extractor.tokenizer(
            [sentence1, sentence2],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        attn_mask = batch_tokens.attention_mask  # (2, L_max)
        s1_len = int(attn_mask[0].sum().item())
        s2_len = int(attn_mask[1].sum().item())

        # 仅取非 padding 部分的 token id 做 diff 比较
        s1_ids = batch_tokens.input_ids[0, :s1_len].tolist()
        s2_ids = batch_tokens.input_ids[1, :s2_len].tolist()

        # 解码为 token 字符串（用于记录）
        s1_tokens = extractor.tokenizer.convert_ids_to_tokens(s1_ids)
        s2_tokens = extractor.tokenizer.convert_ids_to_tokens(s2_ids)

        # ── 找出差异区域 ──
        diff_regions = get_diff(s1_ids, s2_ids)

        if not diff_regions:
            # 两句分词完全一致 → 所有层相似度直接为 1.0
            cosine_sim_matrix[idx, :] = 1.0
            if args.save_diff_vectors:
                diff_vector_dataset.append(
                    build_diff_vector_record(
                        idx, record, s1_ids, s2_ids, s1_tokens, s2_tokens,
                        diff_regions, None, num_layers,
                    )
                )
            continue

        # ── 收集差异区域索引 ──
        s1_diff_indices = collect_diff_indices(diff_regions, side=0)
        s2_diff_indices = collect_diff_indices(diff_regions, side=1)

        if not s1_diff_indices or not s2_diff_indices:
            # 某一句没有差异 token（如纯插入/删除）→ 相似度为 0.0
            cosine_sim_matrix[idx, :] = 0.0
            if args.save_diff_vectors:
                diff_vector_dataset.append(
                    build_diff_vector_record(
                        idx, record, s1_ids, s2_ids, s1_tokens, s2_tokens,
                        diff_regions, None, num_layers,
                    )
                )
            continue

        # ── 批量编码：一次前向传播同时处理两句 ──
        with torch.no_grad():
            hs = extractor.encode(batch_tokens).hidden_states  # tuple of (2, L_max, D)

        # ── 逐层计算差异区域均值向量的余弦相似度 ──
        for layer in range(num_layers):
            layer_hs = hs[layer]  # (2, L_max, D)
            s1_diff = layer_hs[0, s1_diff_indices, :]  # (n1_diff, D)
            s2_diff = layer_hs[1, s2_diff_indices, :]  # (n2_diff, D)

            s1_mean = s1_diff.mean(dim=0)  # (D,)
            s2_mean = s2_diff.mean(dim=0)  # (D,)

            # 使用 PyTorch 内置余弦相似度，并用 nan_to_num 防御零向量
            sim = F.cosine_similarity(
                s1_mean.unsqueeze(0), s2_mean.unsqueeze(0)
            )
            cosine_sim_matrix[idx, layer] = torch.nan_to_num(
                sim, nan=0.0
            ).item()

        # ── 收集差异向量数据 ──
        if args.save_diff_vectors:
            diff_vector_dataset.append(
                build_diff_vector_record(
                    idx, record, s1_ids, s2_ids, s1_tokens, s2_tokens,
                    diff_regions, hs, num_layers,
                )
            )

        # 显式释放 GPU 显存，避免随循环累积
        del hs, batch_tokens

    # ── 保存相似度矩阵 ──
    df = pd.DataFrame(
        cosine_sim_matrix,
        columns=[f"layer_{i}" for i in range(num_layers)],
    )
    sim_path = os.path.join(args.output_dir, f"spatial_similarities_{args.model_alias}.csv")
    df.to_csv(sim_path, index=False)
    print(f"[信息] 已保存空间相似度矩阵到 {sim_path}")

    # ── 计算每层的准确率与相关性 ──
    acc_res = []
    for layer in range(num_layers):
        layer_similarities = cosine_sim_matrix[:, layer]
        predictions = (layer_similarities > 0.5).astype(int)
        accuracy = np.mean(predictions == labels)
        spearman = scipy.stats.spearmanr(layer_similarities, labels)
        pearson = scipy.stats.pearsonr(layer_similarities, labels)
        acc_res.append({
            "layer": layer,
            "accuracy": accuracy,
            "spearman_corr": spearman.statistic,
            "spearman_pvalue": spearman.pvalue,
            "pearson_corr": pearson.statistic,
            "pearson_pvalue": pearson.pvalue,
        })

    df_acc = pd.DataFrame(acc_res)
    res_path = os.path.join(
        args.output_dir,
        f"spatial_similarity_results_{args.model_alias}.csv",
    )
    df_acc.to_csv(res_path, index=False)
    print(f"[信息] 已保存相似度结果到 {res_path}")

    # ── 保存差异向量数据集 ──
    if args.save_diff_vectors:
        vec_path = os.path.join(
            args.output_dir,
            f"spatial_diff_vectors_{args.model_alias}.pkl",
        )
        with open(vec_path, "wb") as f:
            pickle.dump(diff_vector_dataset, f)
        print(f"[信息] 已保存差异向量数据集到 {vec_path} "
              f"({len(diff_vector_dataset)} 条记录)")


if __name__ == "__main__":
    main()
