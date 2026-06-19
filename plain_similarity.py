# encoding: utf8

import read_data
import extract_hidden_info

import scipy.stats
import torch
import numpy as np
import pandas as pd
import os
import argparse

STRATEGIES = ["mean", "last_token", "eos_token", "max", "weighted_mean", "cls_token"]
"""句向量池化策略"""

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

def main():
    argparser = argparse.ArgumentParser(description="一般计算相似度的脚本")
    argparser.add_argument("--data_file", help="数据文件路径")
    argparser.add_argument("--model_name_or_path", help="模型路径或名称")
    argparser.add_argument("--model_alias", help="用于记录的模型名称")
    argparser.add_argument("--output_dir", default=".", help="输出目录,默认为当前目录")

    args = argparser.parse_args()

    # 初始化隐藏信息提取器
    extractor = extract_hidden_info.HiddenInfoExtractor(args.model_name_or_path, device=DEVICE, dtype=DTYPE)
    # 读取数据
    data = read_data.SpatialDataset(args.data_file)

    results = []
    labels = data.labels  # 标签在外层提取，避免内层循环重复计算

    for strategy in STRATEGIES:
        print(f"[信息] 模型：{args.model_alias}, 池化策略：{strategy}")
        # 应用不同的池化策略，获得相似度的矩阵，矩阵的每一行代表一个句子对，每一列代表一个层的相似度
        similarities = [extractor.similarity(sentence1, sentence2, pooling=strategy) for (sentence1, sentence2) in data.sentences]
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
        df_path = f"{args.output_dir}/similarities_{args.model_alias}_{strategy}.csv"
        df.to_csv(df_path, index=False)
        print(f"[信息] 已保存相似度矩阵到 {df_path}")

    # 2. 将准确率、Spearman相关系数和Pearson相关系数保存在CSV文件中
    df_results = pd.DataFrame(results)
    df_results_path = f"{args.output_dir}/results_{args.model_alias}.csv"
    df_results.to_csv(df_results_path, index=False)
    print(f"[信息] 已保存结果到 {df_results_path}")

if __name__ == "__main__":
    main()