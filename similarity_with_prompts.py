# encoding: utf8

import read_data
import extract_hidden_info

import scipy.stats
import torch
import numpy as np
import pandas as pd
import os
import argparse

STRATEGIES = ["mean", "last_token", "eos_token", "max", "weighted_mean"]
"""句向量池化策略"""

PROMPTS = [
    r'用一个词表示句子"{sentence1}"的意思是',
    r'用一个词表示句子"{sentence2}"的意思是',
    # r'句子"{sentence1}"中，"{target1}"相对于"{reference1}"的空间关系是',
    # r'句子"{sentence2}"中，"{target2}"相对于"{reference2}"的空间关系是',
]

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def add_prompt(record: read_data.SentencePairRecord, prompt_idx: int = 0) -> str:
    """用 record 中的字段替换 prompt 模板里的占位符，返回格式化后的字符串。

    Args:
        record: 一条句子对记录（SentencePairRecord）
        prompt_idx: PROMPTS 中模板的索引

    Returns:
        替换占位符后的 prompt 字符串
    """
    assert 0 <= prompt_idx < len(PROMPTS), "prompt的索引值超出范围"
    record_dict = {
        "sentence1": record.sentence1,
        "sentence2": record.sentence2,
        "target1": record.target1,
        "reference1": record.reference1,
        "target2": record.target2,
        "reference2": record.reference2,
        "pair": record.pair,
        "relation": record.relation,
        "label": record.label,
        "id": record.id,
    }
    return PROMPTS[prompt_idx].format(**record_dict)


def main():
    argparser = argparse.ArgumentParser(description="一般计算相似度的脚本")
    argparser.add_argument("--data_file", help="数据文件路径")
    argparser.add_argument("--model_name_or_path", help="模型路径或名称")
    argparser.add_argument("--model_alias", help="用于记录的模型名称")
    argparser.add_argument("--output_dir", default=".", help="输出目录,默认为当前目录")

    args = argparser.parse_args()

    # 初始化隐藏信息提取器
    extractor = extract_hidden_info.HiddenInfoExtractor(args.model_name_or_path, device=DEVICE, torch_dtype=DTYPE)
    # 读取数据
    data = read_data.SpatialDataset(args.data_file)

    results = []
    labels = data.labels  # 标签在外层提取，避免内层循环重复计算

    assert len(data) % 2 == 0, "数据长度必须是偶数"
    for p in range(0, len(prompted_sentences), 2):
        # 为每个句子添加prompt信息
        prompted_sentences = [(add_prompt(rec, p), add_prompt(rec, p + 1)) for rec in data]
        for strategy in STRATEGIES:
            print(f"[信息] 模型：{args.model_alias}, 池化策略：{strategy}")
            # 应用不同的池化策略，获得相似度的矩阵，矩阵的每一行代表一个句子对，每一列代表一个层的相似度
            similarities = [extractor.similarity(sentence1, sentence2, pooling=strategy, show_progress=False) for (sentence1, sentence2) in prompted_sentences]
            similarities_by_pair_layer = [sim["layer_similarities"] for sim in similarities]
            similarities_matrix = np.array(similarities_by_pair_layer)

            # 计算每层的准确率和与标签的Spearman相关系数、Pearson相关系数
            for i in range(similarities_matrix.shape[1]):
                layer_similarities = similarities_matrix[:, i]
                # 假设相似度大于0.5的为正例，小于等于0.5的为负例
                predictions = (layer_similarities > 0.5).astype(int)
                accuracy = np.mean(predictions == labels)
                spearman = scipy.stats.spearmanr(layer_similarities, labels)
                pearson = scipy.stats.pearsonr(layer_similarities, labels)
                results.append({
                    "prompt_group": p, 
                    "strategy": strategy,
                    "layer": i,
                    "accuracy": accuracy,
                    "spearman_corr": spearman.statistic,
                    "spearman_pvalue": spearman.pvalue,
                    "pearson_corr": pearson.statistic,
                    "pearson_pvalue": pearson.pvalue,
                })

            # 存储结果
            # 1. 将相似度的矩阵保存在CSV文件中
            df = pd.DataFrame(similarities_matrix)
            df_path = f"{args.output_dir}/similarities_{args.model_alias}_prompt{p}_{strategy}.csv"
            df.to_csv(df_path, index=False)
            print(f"[信息] 已保存相似度矩阵到 {df_path}")

    # 2. 将准确率、Spearman相关系数和Pearson相关系数保存在CSV文件中
    df_results = pd.DataFrame(results)
    df_results_path = f"{args.output_dir}/results_{args.model_alias}.csv"
    df_results.to_csv(df_results_path, index=False)
    print(f"[信息] 已保存结果到 {df_results_path}")

if __name__ == "__main__":
    main()