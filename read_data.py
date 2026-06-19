# encoding: utf8
"""
数据集读取模块 —— 读取空间语义相似度数据集，提供句子对迭代器。

支持两种 JSON 数组格式：
1. 原始数据集（spatial_dataset.json）：
   [{id, sentence1, sentence2, label, pair, relation}, ...]

2. 标注数据集（含 annotation 字段）：
   [{id, sentence1, sentence2, label, pair, relation,
     annotation: {sentence1: {target, reference},
                  sentence2: {target, reference}}}, ...]
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Iterator


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class TargetReference:
    """句子中提取的目标物与参照物。"""
    target: str
    reference: str

    @classmethod
    def from_dict(cls, d: dict) -> "TargetReference":
        """从字典构建。"""
        return cls(
            target=d.get("target", ""),
            reference=d.get("reference", ""),
        )


@dataclass
class Annotation:
    """一个句子对的空间标注（目标物 & 参照物）。"""
    sentence1: TargetReference = field(default_factory=lambda: TargetReference("", ""))
    sentence2: TargetReference = field(default_factory=lambda: TargetReference("", ""))

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        """从字典构建。"""
        s1 = d.get("sentence1") or {}
        s2 = d.get("sentence2") or {}
        return cls(
            sentence1=TargetReference.from_dict(s1),
            sentence2=TargetReference.from_dict(s2),
        )


@dataclass
class SentencePairRecord:
    """一条完整的句子对数据记录。

    Attributes:
        index:      在数据集中的序号（0-based）
        id:         原始数据中的 id（从 1 开始）
        sentence1:  句子 1 全文
        sentence2:  句子 2 全文
        label:      二分类标签（0 或 1）
        pair:       空间词对，如 "上边-后边"
        relation:   空间关系类别，如 "空间图式交集"、"方向图式重叠"、"涉及多个参照"
        annotation: 空间标注信息（目标物 & 参照物），仅标注数据集有
    """
    index: int
    id: int
    sentence1: str
    sentence2: str
    label: int
    pair: str
    relation: str
    annotation: Optional[Annotation] = None

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def target1(self) -> str:
        """句子 1 中差异部分的目标物。"""
        return self.annotation.sentence1.target if self.annotation else ""

    @property
    def reference1(self) -> str:
        """句子 1 中差异部分的参照物。"""
        return self.annotation.sentence1.reference if self.annotation else ""

    @property
    def target2(self) -> str:
        """句子 2 中差异部分的目标物。"""
        return self.annotation.sentence2.target if self.annotation else ""

    @property
    def reference2(self) -> str:
        """句子 2 中差异部分的参照物。"""
        return self.annotation.sentence2.reference if self.annotation else ""

    @property
    def has_annotation(self) -> bool:
        """是否包含有效的空间标注。"""
        return self.annotation is not None

    @property
    def targets_match(self) -> Optional[bool]:
        """两句的目标物是否相同。无标注时返回 None。"""
        if self.annotation is None:
            return None
        return self.target1 == self.target2

    @property
    def references_match(self) -> Optional[bool]:
        """两句的参照物是否相同。无标注时返回 None。"""
        if self.annotation is None:
            return None
        return self.reference1 == self.reference2


# ---------------------------------------------------------------------------
# 数据集类
# ---------------------------------------------------------------------------

class SpatialDataset:
    """空间语义相似度数据集读取器。

    自动识别：
    - 仅含 label/pair/relation 的原始数据集
    - 含 annotation 字段的标注数据集

    Usage:
        >>> ds = SpatialDataset("spatial_dataset.json")
        >>> len(ds)
        1100
        >>> for rec in ds:
        ...     print(rec.sentence1, rec.sentence2, rec.label)

        >>> ds = SpatialDataset("annotated.json")
        >>> for rec in ds:
        ...     print(rec.target1, rec.reference1)   # 便捷属性
        ...     print(rec.annotation.sentence1.target)  # 完整访问
    """

    def __init__(self, file_path: str):
        """加载 JSON 数据集文件。

        Args:
            file_path: JSON 文件路径。

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 解析失败
            ValueError: JSON 格式不符合预期
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(
                f"期望 JSON 数组格式，实际为: {type(raw).__name__}。"
                "请确认文件格式与 spatial_dataset.json 一致。"
            )

        self._raw = raw
        self._records: list[SentencePairRecord] = [self._parse_item(i, item) for i, item in enumerate(raw)]
        self._has_annotation = any(r.has_annotation for r in self._records)

    # ------------------------------------------------------------------
    # 内部解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_item(index: int, item: dict) -> SentencePairRecord:
        """解析单条 JSON 记录为 SentencePairRecord。"""
        # 仅当 annotation 字段实际存在且非空时才构造 Annotation 对象
        ann_raw = item.get("annotation")
        annotation = Annotation.from_dict(ann_raw) if ann_raw is not None else None

        return SentencePairRecord(
            index=index,
            id=item.get("id", index + 1),
            sentence1=item["sentence1"],
            sentence2=item["sentence2"],
            label=item["label"],
            pair=item.get("pair", ""),
            relation=item.get("relation", ""),
            annotation=annotation,
        )

    # ------------------------------------------------------------------
    # 迭代 & 索引
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[SentencePairRecord]:
        """逐条迭代句子对记录。"""
        return iter(self._records)

    def __len__(self) -> int:
        """返回数据集中的句子对数量。"""
        return len(self._records)

    def __getitem__(self, index: int) -> SentencePairRecord:
        """按索引获取单条记录。支持负索引。"""
        return self._records[index]

    def __repr__(self) -> str:
        n = len(self._records)
        flag = "label+annotation" if self._has_annotation else "label"
        return f"<SpatialDataset: {n} records, fields={flag}>"

    # ------------------------------------------------------------------
    # 便捷批量访问
    # ------------------------------------------------------------------

    @property
    def sentences(self) -> list[tuple[str, str]]:
        """返回所有句子对 [(s1, s2), ...]。"""
        return [(r.sentence1, r.sentence2) for r in self._records]

    @property
    def labels(self) -> list[int]:
        """返回所有标签列表。"""
        return [r.label for r in self._records]

    @property
    def pairs(self) -> list[str]:
        """返回所有空间词对列表。"""
        return [r.pair for r in self._records]

    @property
    def relations(self) -> list[str]:
        """返回所有空间关系类别列表。"""
        return [r.relation for r in self._records]

    @property
    def targets(self) -> list[tuple[str, str]]:
        """返回所有 (target1, target2) 对。无标注时对应位置为空字符串。"""
        return [(r.target1, r.target2) for r in self._records]

    @property
    def references(self) -> list[tuple[str, str]]:
        """返回所有 (reference1, reference2) 对。无标注时对应位置为空字符串。"""
        return [(r.reference1, r.reference2) for r in self._records]

    # ------------------------------------------------------------------
    # 便捷迭代方法
    # ------------------------------------------------------------------

    def iter_sentence_pairs(self) -> Iterator[tuple[str, str, int]]:
        """迭代 (sentence1, sentence2, label) 三元组。"""
        for r in self._records:
            yield r.sentence1, r.sentence2, r.label

    def iter_annotated(self) -> Iterator[tuple[str, str, int, str, str, str, str]]:
        """迭代带标注的七元组。

        Yields:
            (sentence1, sentence2, label, target1, reference1, target2, reference2)
            无标注时 target/reference 为空字符串。
        """
        for r in self._records:
            yield (r.sentence1, r.sentence2, r.label,
                   r.target1, r.reference1, r.target2, r.reference2)

    # ------------------------------------------------------------------
    # 筛选 & 统计
    # ------------------------------------------------------------------

    def filter_by_label(self, label: int) -> list[SentencePairRecord]:
        """按标签筛选记录。"""
        return [r for r in self._records if r.label == label]

    def filter_by_relation(self, relation: str) -> list[SentencePairRecord]:
        """按空间关系类别筛选记录。"""
        return [r for r in self._records if r.relation == relation]

    def filter_annotated(self) -> list[SentencePairRecord]:
        """筛选出有有效标注的记录（annotation 不为 None）。"""
        return [r for r in self._records if r.has_annotation]

    def stats(self) -> dict:
        """返回数据集基本统计信息。"""
        n = len(self._records)
        n_pos = sum(1 for r in self._records if r.label == 1)
        n_neg = sum(1 for r in self._records if r.label == 0)
        n_annotated = sum(1 for r in self._records if r.has_annotation)

        # 空间关系类别分布
        relation_counts: dict[str, int] = {}
        for r in self._records:
            rel = r.relation
            if rel:
                relation_counts[rel] = relation_counts.get(rel, 0) + 1

        # 目标物/参照物匹配统计（仅标注数据）
        n_target_match = sum(1 for r in self._records if r.targets_match is True)
        n_ref_match = sum(1 for r in self._records if r.references_match is True)

        return {
            "total": n,
            "positive": n_pos,
            "negative": n_neg,
            "positive_ratio": n_pos / n if n else 0,
            "annotated": n_annotated,
            "relation_distribution": relation_counts,
            "target_match_count": n_target_match,
            "reference_match_count": n_ref_match,
        }


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def load_dataset(file_path: str) -> SpatialDataset:
    """加载空间语义相似度数据集（SpatialDataset 的工厂函数）。

    Args:
        file_path: JSON 文件路径。

    Returns:
        SpatialDataset 实例。

    Example:
        >>> ds = load_dataset("spatial_dataset.json")
        >>> len(ds)
        1100
    """
    return SpatialDataset(file_path)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_files = [
        "spatial_dataset.json",
        "spatial_info_annotation/spatial_dataset_deepseek-v4-pro_20260618-143434.json",
    ]

    for fp in test_files:
        if not os.path.exists(fp):
            print(f"[SKIP] 文件不存在: {fp}\n")
            continue

        print(f"{'=' * 60}")
        print(f"测试文件: {fp}")
        print(f"{'=' * 60}")

        ds = SpatialDataset(fp)
        print(ds)

        # 统计
        s = ds.stats()
        print(f"\n统计信息:")
        print(f"  总数: {s['total']}")
        print(f"  正例: {s['positive']}, 负例: {s['negative']}")
        print(f"  正例比例: {s['positive_ratio']:.3f}")
        print(f"  已标注: {s['annotated']}")
        print(f"  关系分布: {s['relation_distribution']}")
        if s['annotated'] > 0:
            print(f"  目标物匹配: {s['target_match_count']}")
            print(f"  参照物匹配: {s['reference_match_count']}")

        # 前 3 条详细打印
        print(f"\n--- 前 3 条记录 ---")
        for rec in ds[:3]:
            print(f"\n  [id={rec.id}] label={rec.label}  pair={rec.pair}  relation={rec.relation}")
            print(f"  s1: {rec.sentence1[:80]}...")
            print(f"  s2: {rec.sentence2[:80]}...")
            if rec.has_annotation:
                print(f"  target1={rec.target1!r}  ref1={rec.reference1!r}")
                print(f"  target2={rec.target2!r}  ref2={rec.reference2!r}")
                print(f"  targets_match={rec.targets_match}  refs_match={rec.references_match}")
            else:
                print(f"  (无标注)")

        # 测试迭代
        count = sum(1 for _ in ds)
        assert count == len(ds), f"迭代计数不一致: {count} vs {len(ds)}"
        print(f"\n迭代测试通过: {count} 条")

        # 测试筛选
        if ds._has_annotation:
            annotated = ds.filter_annotated()
            print(f"有效标注记录数: {len(annotated)}")

        # 测试便捷迭代
        for s1, s2, label, t1, r1, t2, r2 in ds.iter_annotated():
            if label == 1:
                break
        print(f"iter_annotated 示例: label={label}, t1={t1!r}, r1={r1!r}")

    print(f"\n{'=' * 60}")
    print("全部测试完成")
