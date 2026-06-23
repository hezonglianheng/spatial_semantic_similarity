# encoding: utf-8
"""对词向量或句向量进行 PCA 降维与可视化。

本模块提供两个核心函数：
- pca_reduce:  将高维向量降到 2 维或 3 维
- plot_pca:    对降维后的向量绘制散点图（2D 或 3D）

用法示例：
    >>> import numpy as np
    >>> from pca_analysis import pca_reduce, plot_pca
    >>> # 假设有 100 个 4096 维的句向量
    >>> vectors = np.random.randn(100, 4096)
    >>> labels = np.array([0, 1, 0, 1, ...])  # 二分类标签
    >>> reduced = pca_reduce(vectors, n_components=2)
    >>> plot_pca(reduced, labels=labels, title="PCA of Sentence Embeddings")

也可以直接传入 extract_hidden_info.HiddenInfoExtractor.embed() 的三维输出
（num_layers, num_sentences, hidden_dim），指定 layer 后自动切片：
    >>> reduced = pca_reduce(embeddings, n_components=2, layer=-1)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3D 投影
from typing import Optional, Union, Literal


# ──────────────────────────────────────────────────────────────────
# 类型别名
# ──────────────────────────────────────────────────────────────────

NDimChoice = Literal[2, 3]
"""PCA 目标维度：仅支持 2 或 3。"""


# ──────────────────────────────────────────────────────────────────
# PCA 降维
# ──────────────────────────────────────────────────────────────────

def pca_reduce(
    vectors: np.ndarray,
    n_components: NDimChoice = 2,
    *,
    layer: Optional[int] = None,
    scale: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """对高维向量做 PCA 降维，返回降维后的坐标与 PCA 对象。

    支持两种输入形状：
      - 2D: (n_samples, n_features) —— 例如从 JSON 加载的词向量列表
      - 3D: (n_layers, n_samples, n_features) —— 例如
        ``HiddenInfoExtractor.embed()`` 的输出。此时需要通过 ``layer``
        参数指定要使用的层索引。

    Args:
        vectors:       高维向量数组。
        n_components:  目标维度，2 或 3。
        layer:         当 vectors 为 3D 时，指定要降维的层索引。
                       0 表示 embedding 层，-1 表示最后一层。
                       若 vectors 为 2D，此参数被忽略。
        scale:         是否在 PCA 前做 StandardScaler 标准化。
                       推荐保持 True，避免大方差维度主导主成分。
        random_state:  PCA 随机种子（当使用随机化 SVD 时有效）。

    Returns:
        (reduced, pca):
          - reduced:  shape (n_samples, n_components)，降维后的坐标
          - pca:      已拟合的 sklearn PCA 对象，可查询
                      ``explained_variance_ratio_`` 等信息。

    Raises:
        ValueError: n_components 不是 2 或 3，或 vectors 维度异常。

    Example:
        >>> # 2D 输入
        >>> vecs = np.random.randn(200, 4096)
        >>> coords, pca = pca_reduce(vecs, n_components=2)
        >>> print(coords.shape)  # (200, 2)
        >>> print(pca.explained_variance_ratio_)

        >>> # 3D 输入（多层的句嵌入）
        >>> emb = np.random.randn(33, 200, 4096)  # 33 层, 200 句
        >>> coords, pca = pca_reduce(emb, n_components=2, layer=-1)
    """
    if n_components not in (2, 3):
        raise ValueError(
            f"n_components 必须为 2 或 3，实际为 {n_components}"
        )

    # ── 形状归一化 ──
    if vectors.ndim == 3:
        if layer is None:
            raise ValueError(
                "输入为 3D (n_layers, n_samples, n_features)，"
                "请通过 layer 参数指定要使用的层（例如 layer=-1 表示最后一层）。"
            )
        vectors = vectors[layer, :, :]
    elif vectors.ndim == 2:
        pass
    else:
        raise ValueError(
            f"vectors 维度必须为 2D 或 3D，实际为 {vectors.ndim}D，"
            f"shape = {vectors.shape}"
        )

    if vectors.shape[0] < 2:
        raise ValueError(
            f"至少需要 2 个样本才能做 PCA，当前样本数为 {vectors.shape[0]}"
        )

    # ── 可选标准化 ──
    if scale:
        vectors = StandardScaler().fit_transform(vectors)

    # ── PCA ──
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(vectors)

    return reduced, pca


# ──────────────────────────────────────────────────────────────────
# PCA 可视化
# ──────────────────────────────────────────────────────────────────

def _validate_colors_and_labels(
    labels: Optional[np.ndarray],
    n_samples: int,
) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """校验 labels 参数并提取类别列表。"""
    if labels is None:
        return None, None
    labels = np.asarray(labels)
    if len(labels) != n_samples:
        raise ValueError(
            f"labels 长度 ({len(labels)}) 与样本数 ({n_samples}) 不一致"
        )
    unique = sorted(set(labels))
    return labels, [str(u) for u in unique]


def plot_pca(
    reduced: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[dict] = None,
    title: str = "PCA Projection",
    alpha: float = 0.7,
    s: Union[float, list[float]] = 30,
    cmap: str = "tab10",
    figsize: tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """对 PCA 降维结果绘制 2D 或 3D 散点图。

    根据 reduced 的列数自动选择 2D 平面图或 3D 立体图。

    Args:
        reduced:     PCA 降维后的坐标，shape (n_samples, 2) 或 (n_samples, 3)。
        labels:      每个样本的类别标签（整数或字符串），用于按颜色区分。
                     为 None 时所有点使用同一颜色。
        label_names: 标签值到显示名称的映射，例如 {0: "正例", 1: "负例"}。
                     未映射的标签直接使用其字符串形式。
        title:       图表标题。
        alpha:       点的透明度。
        s:           点的大小，可为单一值或与样本数等长的列表。
        cmap:        颜色映射名称（当 labels 为数值时使用）。
        figsize:     图像尺寸 (width, height)。
        save_path:   若提供，将图像保存到该路径。
        dpi:         保存图像的分辨率。
        show:        是否调用 ``plt.show()`` 显示图像。
        ax:          可选，在已有的 Axes 上绘制（仅支持 2D）。

    Returns:
        matplotlib Figure 对象。

    Example:
        >>> coords, pca = pca_reduce(vectors, n_components=2)
        >>> plot_pca(coords, labels=labels, title="2D PCA")
        >>> # 3D 示例
        >>> coords3d, _ = pca_reduce(vectors, n_components=3)
        >>> plot_pca(coords3d, labels=labels, title="3D PCA")
    """
    if reduced.ndim != 2:
        raise ValueError(
            f"reduced 必须为 2D 数组，实际为 {reduced.ndim}D"
        )
    n_components = reduced.shape[1]
    if n_components not in (2, 3):
        raise ValueError(
            f"reduced 的列数必须为 2 或 3，实际为 {n_components}"
        )

    n_samples = reduced.shape[0]
    labels, unique_labels = _validate_colors_and_labels(labels, n_samples)

    # ── 推断是否在外部 Axes 上绘制 ──
    external_ax = ax is not None
    if external_ax and n_components != 2:
        raise ValueError("外部 Axes 仅支持 2D 降维结果")

    # ── 创建 Figure ──
    if external_ax:
        fig = ax.figure  # type: ignore[union-attr]
    else:
        subplot_kw: dict = {}
        if n_components == 3:
            subplot_kw = {"projection": "3d"}
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)  # type: ignore[arg-type]

    is_3d = n_components == 3

    # ── 绘制 ──
    if labels is None:
        # 无标签 → 单一颜色
        if is_3d:
            ax.scatter(  # type: ignore[attr-defined]
                reduced[:, 0], reduced[:, 1], reduced[:, 2],
                alpha=alpha, s=s, c="steelblue", edgecolors="none",
            )
        else:
            ax.scatter(  # type: ignore[attr-defined]
                reduced[:, 0], reduced[:, 1],
                alpha=alpha, s=s, c="steelblue", edgecolors="none",
            )
    else:
        # 检查 labels 是否为纯数字（用于 colormap）
        try:
            label_vals = labels.astype(float)
            numeric_labels = True
        except (ValueError, TypeError):
            label_vals = None  # type: ignore[assignment]
            numeric_labels = False

        if numeric_labels and label_vals is not None:
            # 数值标签 → 使用 colormap
            if is_3d:
                sc = ax.scatter(  # type: ignore[attr-defined]
                    reduced[:, 0], reduced[:, 1], reduced[:, 2],
                    c=label_vals, cmap=cmap, alpha=alpha, s=s,
                    edgecolors="none",
                )
                cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
                cbar.set_label("Label")
            else:
                sc = ax.scatter(  # type: ignore[attr-defined]
                    reduced[:, 0], reduced[:, 1],
                    c=label_vals, cmap=cmap, alpha=alpha, s=s,
                    edgecolors="none",
                )
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("Label")
        else:
            # 字符串/离散标签 → 逐类绘制 + 图例
            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                display_name = (
                    label_names.get(lbl, str(lbl))
                    if label_names
                    else str(lbl)
                )
                if is_3d:
                    ax.scatter(  # type: ignore[attr-defined]
                        reduced[mask, 0],
                        reduced[mask, 1],
                        reduced[mask, 2],
                        label=display_name,
                        alpha=alpha,
                        s=s,
                        edgecolors="none",
                    )
                else:
                    ax.scatter(  # type: ignore[attr-defined]
                        reduced[mask, 0],
                        reduced[mask, 1],
                        label=display_name,
                        alpha=alpha,
                        s=s,
                        edgecolors="none",
                    )
            ax.legend(loc="best")  # type: ignore[attr-defined]

    # ── 轴标签与标题 ──
    var_labels = ["PC1", "PC2", "PC3"]
    if is_3d:
        ax.set_xlabel(var_labels[0])  # type: ignore[attr-defined]
        ax.set_ylabel(var_labels[1])  # type: ignore[attr-defined]
        ax.set_zlabel(var_labels[2])  # type: ignore[attr-defined]
    else:
        ax.set_xlabel(var_labels[0])  # type: ignore[attr-defined]
        ax.set_ylabel(var_labels[1])  # type: ignore[attr-defined]

    ax.set_title(title)  # type: ignore[attr-defined]

    # ── 保存 ──
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[信息] 图像已保存至 {save_path}")

    if show and not external_ax:
        plt.show()

    return fig


# ──────────────────────────────────────────────────────────────────
# 便捷函数：从 extract_hidden_info 的句嵌入直接 PCA + 绘图
# ──────────────────────────────────────────────────────────────────

def pca_plot_from_embeddings(
    embeddings: np.ndarray,
    n_components: NDimChoice = 2,
    *,
    layer: int = -1,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[dict] = None,
    scale: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **plot_kwargs,
) -> tuple[np.ndarray, PCA, plt.Figure]:
    """一站式函数：从句嵌入到 PCA 降维、绘图。

    整合 pca_reduce() 与 plot_pca()，适合快速探索。

    Args:
        embeddings:   3D 句嵌入 (n_layers, n_sentences, hidden_dim)
                      或 2D 向量 (n_samples, n_features)。
        n_components: 目标维度 2 或 3。
        layer:        当 embeddings 为 3D 时使用的层索引（默认 -1 = 最后一层）。
        labels:       样本类别标签。
        label_names:  标签值到显示名称的映射。
        scale:        是否在 PCA 前做 StandardScaler。
        title:        图表标题（为 None 时自动生成）。
        save_path:    图像保存路径。
        show:         是否显示图像。
        **plot_kwargs: 传递给 plot_pca() 的额外参数（alpha, s, cmap, figsize 等）。

    Returns:
        (reduced, pca, fig):
          - reduced: 降维后的坐标
          - pca:     已拟合的 PCA 对象
          - fig:     matplotlib Figure

    Example:
        >>> from extract_hidden_info import HiddenInfoExtractor
        >>> from pca_analysis import pca_plot_from_embeddings
        >>> ext = HiddenInfoExtractor("path/to/model")
        >>> emb = ext.embed(sentences, pooling="mean")
        >>> coords, pca, fig = pca_plot_from_embeddings(
        ...     emb, n_components=2, layer=-1, labels=labels
        ... )
    """
    reduced, pca = pca_reduce(
        embeddings,
        n_components=n_components,
        layer=layer,
        scale=scale,
    )

    if title is None:
        layer_str = f"layer_{layer}" if embeddings.ndim == 3 else "all"
        title = f"PCA ({n_components}D) — {layer_str}"

    fig = plot_pca(
        reduced,
        labels=labels,
        label_names=label_names,
        title=title,
        save_path=save_path,
        show=show,
        **plot_kwargs,
    )

    return reduced, pca, fig


# ──────────────────────────────────────────────────────────────────
# 简洁 API
# ──────────────────────────────────────────────────────────────────

__all__ = [
    "pca_reduce",
    "plot_pca",
    "pca_plot_from_embeddings",
]
