# encoding: utf-8

import extract_similarity_directly as esd
import read_data

import scipy.stats
import pandas as pd
import os
import numpy as np
import argparse

PROMPTS = [
    r'用一个词表示句子"{sentence1}"的意思是',
    r'用一个词表示句子"{sentence2}"的意思是',
]

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

    data = read_data.SpatialDataset(args.data_file)

    extractor = esd.SimilarityExtractor(args.model_name_or_path)

    # 收集所有组的标签（只需计算一次，各组共用）
    labels = [label for label in data.labels]

    assert len(PROMPTS) % 2 == 0, "PROMPTS列表的长度应该是偶数，因为每对句子需要两个prompt"

    # 收集各组结果的列表
    results_summary = []

    for p in range(0, len(PROMPTS), 2):
        prompt_group = (p // 2) + 1  # 用于记录的prompt组编号，从1开始
        pair_with_prompt = [(add_prompt(record, p), add_prompt(record, p + 1)) for record in data]

        similarities = [extractor.similarity(sentence1, sentence2).to("cpu").item() for (sentence1, sentence2) in pair_with_prompt]

        # 保存相似度结果到CSV文件
        df = pd.DataFrame({"sentence1": [s1 for s1, s2 in data.sentences], "sentence2": [s2 for s1, s2 in data.sentences], "similarity": similarities})
        df.to_csv(os.path.join(args.output_dir, f"similarities_prompt{prompt_group}_{args.model_alias}.csv"), index=False)

        # 计算准确率和相关性
        predicted = [1 if sim > 0.5 else 0 for sim in similarities]

        # 计算准确率
        accuracy = np.mean(np.array(predicted) == np.array(labels))

        # 计算Pearson和Spearman相关性
        pearson = scipy.stats.pearsonr(similarities, labels)
        spearman = scipy.stats.spearmanr(similarities, labels)

        results_summary.append({
            "prompt_group": prompt_group,
            "accuracy": accuracy,
            "pearson": pearson,
            "spearman": spearman,
        })

    # 将所有组的结果统一写入同一个文件
    with open(os.path.join(args.output_dir, f"results_prompt_{args.model_alias}.txt"), "w") as f:
        for result in results_summary:
            f.write(f"=== Prompt Group {result['prompt_group']} ===\n")
            f.write(f"Accuracy: {result['accuracy']}\n")
            f.write(f"Pearson correlation: {result['pearson'][0]}\n")
            f.write(f"Pearson correlation R: {result['pearson'][1]}\n")
            f.write(f"Spearman correlation: {result['spearman'][0]}\n")
            f.write(f"Spearman correlation R: {result['spearman'][1]}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
