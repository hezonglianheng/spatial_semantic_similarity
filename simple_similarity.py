# encoding: utf-8

import extract_similarity_directly as esd
import read_data

import scipy.stats
import pandas as pd
import os
import numpy as np
import argparse

def main():
    argparser = argparse.ArgumentParser(description="一般计算相似度的脚本")
    argparser.add_argument("--data_file", help="数据文件路径")
    argparser.add_argument("--model_name_or_path", help="模型路径或名称")
    argparser.add_argument("--model_alias", help="用于记录的模型名称")
    argparser.add_argument("--output_dir", default=".", help="输出目录,默认为当前目录")

    args = argparser.parse_args()

    data = read_data.SpatialDataset(args.data_file)

    extractor = esd.SimilarityExtractor(args.model_name_or_path)

    similarities = [extractor.similarity(sentence1, sentence2).to("cpu").item() for (sentence1, sentence2) in data.sentences]

    # 保存相似度结果到CSV文件
    df = pd.DataFrame({"sentence1": [s1 for s1, s2 in data.sentences], "sentence2": [s2 for s1, s2 in data.sentences], "similarity": similarities})
    df.to_csv(os.path.join(args.output_dir, f"similarities_{args.model_alias}.csv"), index=False)

    # 计算准确率和相关性
    labels = [label for label in data.labels]
    predicted = [1 if sim > 0.5 else 0 for sim in similarities]

    # 计算准确率
    accuracy = np.mean(np.array(predicted) == np.array(labels))

    # 计算Pearson和Spearman相关性
    pearson = scipy.stats.pearsonr(similarities, labels)
    spearman = scipy.stats.spearmanr(similarities, labels)

    # 写入文件
    with open(os.path.join(args.output_dir, f"results_{args.model_alias}.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Pearson correlation: {pearson[0]}\n")
        f.write(f"Pearson correlation R: {pearson[1]}\n")
        f.write(f"Spearman correlation: {spearman[0]}\n")
        f.write(f"Spearman correlation R: {spearman[1]}\n")

if __name__ == "__main__":
    main()
