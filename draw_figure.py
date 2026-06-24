# encoding: utf8

"""
从 CSV 文件中读取数据、按条件筛选行、并绘制折线图。

功能：
  1. 批量读入 CSV 文件，自动合并或分文件保留
  2. 按指定列和条件筛选行 —— 支持数字范围（等于、大于、小于、区间等）
     以及文字/类别匹配
  3. 支持按文件分别筛选：不同 CSV 文件可应用不同的筛选条件
  4. 不同来源文件的数据使用不同颜色绘制在同一图上
  5. 支持图例放置在绘图区域外侧（--legend_outside）
  6. 根据筛选结果绘制折线图，x/y 轴范围由数据自动计算（可加边距）

用法示例：
  # === 命令行 ===
  # 基本用法：多文件合并，按 _source_file 分组，图例在外
  python draw_figure.py \
      --csv_files results_qwen.csv results_llama.csv \
      --filters "accuracy > 0.5" "strategy == mean" \
      --x_column layer --y_columns accuracy spearman_corr \
      --group_by _source_file \
      --legend_outside \
      --output figure.png

  # 按文件分别筛选：不同文件使用不同的筛选条件
  python draw_figure.py \
      --csv_files results_qwen.csv results_llama.csv \
      --file_filters "results_qwen.csv:accuracy > 0.5" \
                      "results_llama.csv:accuracy > 0.7, strategy == mean" \
      --x_column layer --y_columns accuracy \
      --group_by _source_file \
      --legend_outside \
      --output figure.png

  # === Python API ===
  from draw_figure import CSVDataLoader, FilterCondition, LineChartPlotter

  loader = CSVDataLoader(
      ["a.csv", "b.csv"],
      file_filters={
          "a.csv": ["accuracy > 0.5"],
          "b.csv": ["accuracy > 0.7"],
      },
  )
  df = loader.load()

  filtered = FilterCondition.apply(df, [
      ("strategy", "==", "mean"),
  ])

  plotter = LineChartPlotter(filtered)
  plotter.plot(
      x_column="layer",
      y_columns=["accuracy", "spearman_corr"],
      group_by="_source_file",
      legend_outside=True,
      output_path="figure.png",
  )
"""

import os
import argparse
import warnings
from typing import Optional, Union, Sequence, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合脚本/服务器
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 全局绘图样式设置
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# 中文字体配置 —— 直接使用 TTF 字体文件
# ---------------------------------------------------------------------------
#
# 为什么不能只设 font.family = "SimHei"？
# ─────────────────────────────────────
# matplotlib 的字体解析不是按字体名直接查找，而是通过"通用族"体系：
#   font.family = "sans-serif"
#       → 查 font.sans-serif = ["DejaVu Sans", "Arial", ...]  按序尝试
# 把 "SimHei" 直接赋给 font.family 会被当成一个不存在的通用族名，
# matplotlib 找不到后回退到默认的 sans-serif 族 → DejaVu Sans。
#
# 为什么 plt.style.context() 也会导致问题？
# ─────────────────────────────────────────
# 即使使用 "default" 样式，plt.style.context() 也会加载样式中写死的
# font.sans-serif 列表（如 ["DejaVu Sans", ...]），覆盖掉我们的设置。
# 所以必须在样式上下文内部再次应用字体配置。
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
_FONT_NAME = _font_prop.get_name()

# ---- Step 3: 设置 matplotlib 全局字体 ----
plt.rcParams["font.sans-serif"] = [_FONT_NAME] + plt.rcParams["font.sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

print(f"[字体] 加载 TTF: {FONT_TTF_PATH} → {_FONT_NAME}")

# 验证字体已正确注册到 fontManager
_ttf_entries = [f for f in fm.fontManager.ttflist if f.name == _FONT_NAME]
print(f"[字体] fontManager 中匹配 '{_FONT_NAME}' 的条目: {len(_ttf_entries)} 个")


# ---------------------------------------------------------------------------
# 筛选条件解析
# ---------------------------------------------------------------------------

# 支持的操作符及其别名
_OPERATOR_MAP = {
    "==": "eq", "=": "eq", "eq": "eq",
    "!=": "ne", "ne": "ne",
    ">": "gt", "gt": "gt",
    ">=": "ge", "ge": "ge",
    "<": "lt", "lt": "lt",
    "<=": "le", "le": "le",
    "between": "between", "bt": "between", "[]": "between",
    "in": "isin", "isin": "isin",
    "contains": "contains", "has": "contains",
    "startswith": "startswith",
    "endswith": "endswith",
}


def _parse_single_condition(cond_str: str) -> tuple[str, str, Any]:
    """将形如 ``"accuracy > 0.5"`` 或 ``"strategy == mean"`` 的字符串
    解析为 ``(column, operator, value)`` 三元组。

    支持的操作符:
        == =  相等
        !=    不等
        >     大于
        >=    大于等于
        <     小于
        <=    小于等于
        between / []  区间，值用逗号分隔，如 "layer between 0,12"
        in     属于集合，值用逗号分隔，如 "strategy in mean,last_token"
        contains  字符串包含，如 "name contains test"
        startswith / endswith  字符串前缀/后缀

    示例::

        _parse_single_condition("accuracy > 0.5")
        # → ("accuracy", "gt", 0.5)

        _parse_single_condition("strategy == mean")
        # → ("strategy", "eq", "mean")
    """
    tokens = cond_str.strip().split(maxsplit=2)
    if len(tokens) < 3:
        raise ValueError(
            f"无法解析筛选条件: '{cond_str}'。"
            f"格式应为: '<列名> <操作符> <值>'"
        )

    col, op_str, val_str = tokens[0], tokens[1], tokens[2]
    op_str = op_str.lower()

    if op_str not in _OPERATOR_MAP:
        raise ValueError(
            f"不支持的操作符: '{op_str}'。支持: "
            f"{sorted(set(_OPERATOR_MAP.keys()))}"
        )

    op = _OPERATOR_MAP[op_str]

    # 解析值
    if op in ("between",):
        parts = [p.strip() for p in val_str.split(",")]
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            lo, hi = parts[0], parts[1]
        value = (lo, hi)
    elif op in ("isin",):
        parts = [p.strip() for p in val_str.split(",")]
        # 尝试转换为数值
        converted = []
        for p in parts:
            try:
                converted.append(float(p) if "." in p or "e" in p.lower() else int(p))
            except ValueError:
                converted.append(p)
        value = converted
    elif op in ("contains", "startswith", "endswith"):
        value = val_str.strip().strip("'\"")
    else:
        value = val_str.strip().strip("'\"")
        # 尝试转换为数值
        try:
            value = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            pass

    return col, op, value


# ---------------------------------------------------------------------------
# FilterCondition —— 筛选条件对象
# ---------------------------------------------------------------------------

class FilterCondition:
    """表示一组筛选条件，可对 DataFrame 逐行过滤。

    参数:
        conditions: 条件列表。每个元素可以是:
            - ``(column, operator, value)`` 三元组
            - ``"column operator value"`` 字符串

    示例::

        fc = FilterCondition([
            ("accuracy", ">", 0.5),
            "strategy == mean",
            ("layer", "between", (0, 12)),
        ])
        filtered_df = fc.apply(df)
    """

    def __init__(self, conditions: list):
        self._raw = conditions
        self._parsed: list[tuple[str, str, Any]] = []
        for c in conditions:
            if isinstance(c, str):
                self._parsed.append(_parse_single_condition(c))
            elif isinstance(c, (list, tuple)) and len(c) == 3:
                col, op_raw, val = c
                op = _OPERATOR_MAP.get(op_raw.lower(), op_raw.lower())
                self._parsed.append((col, op, val))
            else:
                raise ValueError(f"无法解析条件: {c!r}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 DataFrame 施加所有条件（AND 逻辑），返回筛选后的副本。"""
        mask = pd.Series(True, index=df.index)
        for col, op, val in self._parsed:
            if col not in df.columns:
                raise KeyError(
                    f"列 '{col}' 不存在。可用列: {list(df.columns)}"
                )

            series = df[col]

            if op == "eq":
                m = series == val
            elif op == "ne":
                m = series != val
            elif op == "gt":
                m = series > val
            elif op == "ge":
                m = series >= val
            elif op == "lt":
                m = series < val
            elif op == "le":
                m = series <= val
            elif op == "between":
                lo, hi = val
                m = series.between(lo, hi)
            elif op == "isin":
                m = series.isin(val)
            elif op == "contains":
                m = series.astype(str).str.contains(val, na=False)
            elif op == "startswith":
                m = series.astype(str).str.startswith(val, na=False)
            elif op == "endswith":
                m = series.astype(str).str.endswith(val, na=False)
            else:
                raise ValueError(f"未知操作符: {op}")

            mask = mask & m

        return df.loc[mask].copy()

    def __repr__(self) -> str:
        return f"FilterCondition({self._parsed!r})"


# ---------------------------------------------------------------------------
# CSVDataLoader —— CSV 批量加载器
# ---------------------------------------------------------------------------

class CSVDataLoader:
    """批量读入 CSV 文件，支持添加来源标记和合并。

    参数:
        file_paths:   CSV 文件路径列表（支持 glob 通配符）
        concat:       是否将所有文件合并为一个 DataFrame（默认 True）
        add_file_col: 若为 True 且 concat=True，添加 ``_source_file`` 列标记来源
        csv_kwargs:   传递给 ``pd.read_csv()`` 的额外参数

    示例::

        loader = CSVDataLoader(
            ["results_qwen.csv", "results_llama.csv"],
            add_file_col=True,
        )
        df = loader.load()          # → 合并后的 DataFrame
        dfs = loader.load_separate()  # → dict[str, DataFrame]
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        concat: bool = True,
        add_file_col: bool = True,
        csv_kwargs: Optional[dict] = None,
    ):
        import glob as _glob
        _expanded = []
        for p in file_paths:
            matched = _glob.glob(p)
            if matched:
                _expanded.extend(matched)
            elif os.path.exists(p):
                _expanded.append(p)
            else:
                warnings.warn(f"文件不存在或无法匹配: {p}")

        if not _expanded:
            raise FileNotFoundError(f"没有找到任何可读取的 CSV 文件: {file_paths}")

        self.file_paths = sorted(set(_expanded))
        self.concat = concat
        self.add_file_col = add_file_col and concat
        self.csv_kwargs = csv_kwargs or {}

    def load(self) -> pd.DataFrame:
        """读取所有 CSV 文件并返回（合并或单独的 DataFrame）。"""
        if self.concat:
            return self._load_concat()
        else:
            return self._load_single(self.file_paths[0])

    def load_separate(self) -> dict[str, pd.DataFrame]:
        """分别读取每个文件，返回 ``{文件名: DataFrame}`` 字典。"""
        return {
            os.path.basename(p): self._load_single(p)
            for p in self.file_paths
        }

    def _load_single(self, path: str) -> pd.DataFrame:
        kw = self.csv_kwargs.copy()
        df = pd.read_csv(path, **kw)
        print(f"[加载] {path}  →  {df.shape[0]} 行 × {df.shape[1]} 列")
        return df

    def _load_concat(self) -> pd.DataFrame:
        frames = []
        for path in self.file_paths:
            df = self._load_single(path)
            if self.add_file_col:
                df["_source_file"] = os.path.basename(path)
            frames.append(df)

        result = pd.concat(frames, axis=0, ignore_index=True)
        print(f"[合并] 共 {len(self.file_paths)} 个文件, {result.shape[0]} 行")
        return result


# ---------------------------------------------------------------------------
# LineChartPlotter —— 折线图绘制器
# ---------------------------------------------------------------------------

class LineChartPlotter:
    """根据给定的 DataFrame 绘制折线图。

    支持:
        - 多条 y 列同时绘制（每条一个子图或叠加）
        - 按某列分组，每组画一条线（不同颜色）
        - 自动计算 x/y 轴范围，可加边距比例
        - 自定义标题、标签、图例位置

    参数:
        data:  用于绘图的数据
        style: 预设样式名（"default" / "ggplot" / "seaborn" / "fivethirtyeight"）
    """

    _STYLES = {
        "default": "default",
        "ggplot": "ggplot",
        "seaborn": "seaborn-v0_8",
        "fivethirtyeight": "fivethirtyeight",
    }

    def __init__(self, data: pd.DataFrame, style: str = "default"):
        self.data = data.copy()
        self.style = self._STYLES.get(style, style)

    # ------------------------------------------------------------------
    # 绘图主方法
    # ------------------------------------------------------------------

    def plot(
        self,
        x_column: str,
        y_columns: Union[str, list[str]],
        group_by: Optional[str] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_labels: Optional[Union[str, list[str]]] = None,
        x_padding: float = 0.05,
        y_padding: float = 0.10,
        figsize: tuple[float, float] = (10, 6),
        legend_loc: str = "best",
        legend_outside: bool = False,
        legend_bbox: tuple[float, float] = (1.02, 1.0),
        marker: str = "o",
        markersize: int = 5,
        linewidth: float = 1.8,
        subplots: bool = False,
        colors: Optional[list] = None,
        show_grid: bool = True,
        dpi: int = 150,
    ) -> plt.Figure:
        """绘制折线图。

        参数:
            x_column:      用作 x 轴的列名
            y_columns:     用作 y 轴的列名（一个或多个）
            group_by:      按此列分组，每组画一条折线
            output_path:   若提供，保存图片至此路径（支持 .png / .pdf / .svg）
            title:         图表标题
            x_label:       x 轴标签（默认使用列名）
            y_labels:      y 轴标签（默认使用列名，多条时对应列表）
            x_padding:     x 轴范围边距比例（0.05 = 5%）
            y_padding:     y 轴范围边距比例
            figsize:       图像大小 (宽, 高) 英寸
            legend_loc:    图例位置（legend_outside=True 时仅作参考）
            legend_outside: True 时图例放置在绘图区域外侧
            legend_bbox:   图例外侧放置时的锚点坐标（默认 (1.02, 1.0) 即右上外侧）
            marker:        折线图标记样式
            markersize:    标记大小
            linewidth:     线宽
            subplots:      True 时每条 y 列绘制独立子图
            colors:        自定义颜色列表
            show_grid:     是否显示网格
            dpi:           图像分辨率

        返回:
            matplotlib Figure 对象
        """
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        if isinstance(y_labels, str):
            y_labels = [y_labels]
        if y_labels is None:
            y_labels = y_columns

        # 校验列存在
        _all_cols = [x_column] + y_columns
        if group_by:
            _all_cols.append(group_by)
        missing = [c for c in _all_cols if c not in self.data.columns]
        if missing:
            raise KeyError(f"以下列不存在于数据中: {missing}。"
                           f"可用列: {list(self.data.columns)}")

        # ---- 准备分组 ----
        if group_by:
            groups = self.data.groupby(group_by, sort=True)
            group_names = list(groups.groups.keys())
            n_groups = len(group_names)
        else:
            n_groups = 1

        # ---- 颜色映射：无自定义颜色时使用 tab10 / tab20 调色板 ----
        if colors is None and group_by and n_groups > 1:
            cmap = plt.cm.tab10 if n_groups <= 10 else plt.cm.tab20
            colors = [cmap(i % cmap.N) for i in range(n_groups)]
        elif colors is None:
            colors = [None]  # 让 matplotlib 自行处理

        # ---- 创建图像 ----
        with plt.style.context(self.style):
            # ⚠️ 样式上下文会用样式中写死的 font.sans-serif 覆盖全局设置，
            # 必须在此重新应用中文字体配置（见文件顶部字体配置注释）
            plt.rcParams["font.sans-serif"] = [_FONT_NAME] + plt.rcParams["font.sans-serif"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False

            if subplots and len(y_columns) > 1:
                fig, axes = plt.subplots(
                    len(y_columns), 1,
                    figsize=(figsize[0], figsize[1] * len(y_columns)),
                    sharex=True, dpi=dpi,
                )
                if len(y_columns) == 1:
                    axes = [axes]
            else:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                axes = [ax]

            # ---- 画线 ----
            for ax_idx, (y_col, y_lbl) in enumerate(zip(y_columns, y_labels)):
                ax = axes[ax_idx]

                x_min_global, x_max_global = np.inf, -np.inf
                y_min_global, y_max_global = np.inf, -np.inf

                if group_by:
                    for i, (gname, gdf) in enumerate(groups):
                        color = colors[i % len(colors)] if colors else None
                        legend_label = os.path.splitext(str(gname))[0].split("_")[-1]
                        self._draw_single_line(
                            ax, gdf, x_column, y_col,
                            label=legend_label, color=color,
                            marker=marker, markersize=markersize,
                            linewidth=linewidth,
                        )
                        x_min_global = min(x_min_global, gdf[x_column].min())
                        x_max_global = max(x_max_global, gdf[x_column].max())
                        y_min_global = min(y_min_global, gdf[y_col].min())
                        y_max_global = max(y_max_global, gdf[y_col].max())
                else:
                    self._draw_single_line(
                        ax, self.data, x_column, y_col,
                        label=y_lbl, color=colors[0] if colors else None,
                        marker=marker, markersize=markersize,
                        linewidth=linewidth,
                    )
                    x_min_global = self.data[x_column].min()
                    x_max_global = self.data[x_column].max()
                    y_min_global = self.data[y_col].min()
                    y_max_global = self.data[y_col].max()

                # ---- 轴范围和标签 ----
                self._set_axis_limits(
                    ax, x_min_global, x_max_global, x_padding,
                    y_min_global, y_max_global, y_padding,
                )
                ax.set_xlabel(x_label or x_column)
                ax.set_ylabel(y_lbl)
                if show_grid:
                    ax.grid(True, alpha=0.4, linestyle="--")
                if group_by and ax_idx == 0:
                    if legend_outside:
                        ax.legend(
                            loc="upper left",
                            bbox_to_anchor=legend_bbox,
                            borderaxespad=0.0,
                            framealpha=0.9,
                        )
                    else:
                        ax.legend(loc=legend_loc)

            # ---- 总标题 ----
            if title:
                fig.suptitle(title, fontsize=15, fontweight="bold")
            else:
                y_names = ", ".join(y_columns)
                fig.suptitle(f"{y_names} vs {x_column}", fontsize=15, fontweight="bold")

            if legend_outside and group_by:
                # 为外侧图例腾出右侧空间：right_margin 越小，图例空间越大
                overflow = legend_bbox[0] - 1.0  # e.g. 1.02 → 0.02
                right_margin = max(0.65, 0.84 - overflow * 2.5)
                fig.tight_layout(rect=[0, 0, right_margin, 1])
            else:
                fig.tight_layout()

            # ---- 保存 ----
            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                fig.savefig(output_path, dpi=dpi)
                print(f"[保存] 图片已保存至: {output_path}")

            return fig

    # ------------------------------------------------------------------
    # 便捷绘图：多条线叠加在同一子图
    # ------------------------------------------------------------------

    def plot_overlay(
        self,
        x_column: str,
        y_columns: Union[str, list[str]],
        group_by: Optional[str] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """多条 y 列叠加在同一子图上（非 subplots 模式）。"""
        kwargs.setdefault("subplots", False)
        kwargs.setdefault("figsize", kwargs.get("figsize", (10, 6)))
        return self.plot(
            x_column=x_column,
            y_columns=y_columns,
            group_by=group_by,
            output_path=output_path,
            title=title,
            **kwargs,
        )

    # ==================================================================
    # 内部方法
    # ==================================================================

    @staticmethod
    def _draw_single_line(
        ax: plt.Axes,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        label: str = "",
        color=None,
        marker: str = "o",
        markersize: int = 5,
        linewidth: float = 1.8,
    ):
        """在给定的 Axes 上画一条折线。"""
        # 按 x 排序，保证连线顺序正确
        sorted_df = df.sort_values(by=x_col)
        ax.plot(
            sorted_df[x_col], sorted_df[y_col],
            marker=marker, markersize=markersize,
            linewidth=linewidth,
            label=label,
            color=color,
        )

    @staticmethod
    def _set_axis_limits(
        ax: plt.Axes,
        x_min: float, x_max: float,
        x_padding: float,
        y_min: float, y_max: float,
        y_padding: float,
    ):
        """根据数据范围和 padding 比例设置轴范围。"""
        if np.isfinite(x_min) and np.isfinite(x_max):
            x_range = x_max - x_min
            if x_range == 0:
                x_range = max(abs(x_min) * 0.1, 1.0)
            ax.set_xlim(
                x_min - x_range * x_padding,
                x_max + x_range * x_padding,
            )

        if np.isfinite(y_min) and np.isfinite(y_max):
            y_range = y_max - y_min
            if y_range == 0:
                y_range = max(abs(y_min) * 0.1, 0.1)
            ax.set_ylim(
                y_min - y_range * y_padding,
                y_max + y_range * y_padding,
            )

    # ------------------------------------------------------------------
    # 便捷静态方法：一行出图
    # ------------------------------------------------------------------

    @staticmethod
    def quick_plot(
        df: pd.DataFrame,
        x_column: str,
        y_columns: Union[str, list[str]],
        group_by: Optional[str] = None,
        filters: Optional[list] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """一行代码完成筛选 + 绘图。

        参数:
            df:          数据
            x_column:    x 轴列
            y_columns:   y 轴列
            group_by:    分组列
            filters:     筛选条件列表
            output_path: 保存路径
            **kwargs:    传递给 ``LineChartPlotter.plot()``

        返回:
            Figure 对象

        示例::

            LineChartPlotter.quick_plot(
                df, x_column="layer", y_columns=["accuracy", "spearman_corr"],
                filters=["strategy == mean", "accuracy > 0.5"],
                group_by="prompt_group",
                output_path="out.png",
            )
        """
        if filters:
            fc = FilterCondition(filters)
            df = fc.apply(df)
        plotter = LineChartPlotter(df)
        return plotter.plot(
            x_column=x_column,
            y_columns=y_columns,
            group_by=group_by,
            output_path=output_path,
            **kwargs,
        )


# ===================================================================
# 命令行入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 CSV 文件读取数据、筛选、绘制折线图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法：从多个 CSV 读入，筛选 strategy=mean 的行，按层画准确率折线
  python draw_figure.py --csv_files a.csv b.csv --x layer --y accuracy
      --filters "strategy == mean"

  # 多文件合并 + 按来源文件分组（不同颜色）+ 图例外侧
  python draw_figure.py --csv_files results_qwen.csv results_llama.csv
      --x layer --y accuracy --group_by _source_file --legend_outside
      --output figure.png

  # 多条件筛选 + 按 prompt_group 分组画多条线
  python draw_figure.py --csv_files results.csv --x layer --y spearman_corr
      --filters "strategy == mean" "accuracy > 0.5" --group_by prompt_group

  # 同时画多个 y 列 + 文字筛选 + 保存
  python draw_figure.py --csv_files results.csv --x layer
      --y accuracy spearman_corr pearson_corr
      --filters "strategy in mean,last_token" --output my_figure.png

  # 区间筛选
  python draw_figure.py --csv_files results.csv --x layer --y accuracy
      --filters "layer between 0,24"
        """,
    )

    # ---- 文件参数 ----
    parser.add_argument(
        "--csv_files", "-f", nargs="+", required=True,
        help="CSV 文件路径列表（支持通配符，如 results_*.csv）",
    )
    parser.add_argument(
        "--concat", action="store_true", default=True,
        help="将多个 CSV 文件合并为一个 DataFrame（默认开启）",
    )
    parser.add_argument(
        "--no_concat", dest="concat", action="store_false",
        help="不合并，每个文件单独出图",
    )

    # ---- 筛选参数 ----
    parser.add_argument(
        "--filters", "-c", nargs="*", default=[],
        help=(
            "筛选条件，格式: '<列名> <操作符> <值>'。"
            "支持操作符: == != > >= < <= between in contains startswith endswith。"
            "多个条件以 AND 逻辑组合。"
        ),
    )

    # ---- 绘图参数 ----
    parser.add_argument(
        "--x_column", "-x", type=str, required=True,
        help="用作 x 轴的列名",
    )
    parser.add_argument(
        "--y_columns", "-y", nargs="+", required=True,
        help="用作 y 轴的列名（可多个）",
    )
    parser.add_argument(
        "--group_by", "-g", type=str, default=None,
        help="按此列分组，每组画一条折线",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出图片路径（支持 .png/.pdf/.svg），默认显示图片",
    )
    parser.add_argument(
        "--title", "-t", type=str, default=None,
        help="图表标题",
    )
    parser.add_argument(
        "--x_label", type=str, default=None,
        help="x 轴标签",
    )
    parser.add_argument(
        "--y_labels", nargs="*", default=None,
        help="y 轴标签（多个时与 --y_columns 一一对应）",
    )
    parser.add_argument(
        "--x_padding", type=float, default=0.05,
        help="x 轴范围边距比例（默认 0.05）",
    )
    parser.add_argument(
        "--y_padding", type=float, default=0.10,
        help="y 轴范围边距比例（默认 0.10）",
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=[10, 6],
        help="图像大小 宽 高（英寸），默认 10 6",
    )
    parser.add_argument(
        "--legend_loc", type=str, default="best",
        help="图例位置（默认 best）。当 --legend_outside 时此参数仅作参考",
    )
    parser.add_argument(
        "--legend_outside", action="store_true", default=False,
        help="将图例放置在绘图区域外侧（右上角外侧）",
    )
    parser.add_argument(
        "--legend_bbox", type=float, nargs=2, default=[1.02, 1.0],
        help="图例外侧放置时的锚点坐标 (x y)，默认 1.02 1.0",
    )
    parser.add_argument(
        "--marker", type=str, default="o",
        help="折线图标记样式（默认 o）",
    )
    parser.add_argument(
        "--markersize", type=float, default=5,
        help="标记大小",
    )
    parser.add_argument(
        "--linewidth", type=float, default=1.8,
        help="线宽",
    )
    parser.add_argument(
        "--subplots", action="store_true", default=False,
        help="每条 y 列绘制独立子图",
    )
    parser.add_argument(
        "--no_grid", action="store_true", default=False,
        help="不显示网格",
    )
    parser.add_argument(
        "--style", type=str, default="default",
        choices=["default", "ggplot", "seaborn", "fivethirtyeight"],
        help="绘图样式预设",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="图像 DPI",
    )

    # ---- 数据输出 ----
    parser.add_argument(
        "--print_filtered", action="store_true", default=False,
        help="打印筛选后的数据到终端",
    )
    parser.add_argument(
        "--save_filtered", type=str, default=None,
        help="将筛选后的数据保存为 CSV 文件",
    )

    args = parser.parse_args()

    # ================================================================
    # Step 1: 加载 CSV
    # ================================================================
    loader = CSVDataLoader(
        file_paths=args.csv_files,
        concat=args.concat,
        add_file_col=args.concat,
    )

    if args.concat:
        df = loader.load()
        data_dict = {"merged": df}
    else:
        data_dict = loader.load_separate()

    # ================================================================
    # Step 2: 筛选
    # ================================================================
    if args.filters:
        print(f"\n[筛选] 条件: {args.filters}")
        fc = FilterCondition(args.filters)
        for key in data_dict:
            before = data_dict[key].shape[0]
            data_dict[key] = fc.apply(data_dict[key])
            after = data_dict[key].shape[0]
            print(f"  {key}: {before} → {after} 行")
        print()

    # ================================================================
    # Step 3: 输出筛选结果（可选）
    # ================================================================
    if args.print_filtered:
        for key, d in data_dict.items():
            print(f"--- {key} ---")
            print(d.to_string())
            print()

    if args.save_filtered:
        for key, d in data_dict.items():
            base, ext = os.path.splitext(args.save_filtered)
            path = f"{base}_{key}{ext}" if len(data_dict) > 1 else args.save_filtered
            d.to_csv(path, index=False)
            print(f"[保存数据] {path}")

    # ================================================================
    # Step 4: 绘图
    # ================================================================
    for key, d in data_dict.items():
        if d.empty:
            print(f"[警告] {key} 筛选后无数据，跳过绘图。")
            continue

        output_path = args.output
        if output_path and len(data_dict) > 1:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{key}{ext}"

        title = args.title or f"{key}"

        plotter = LineChartPlotter(d, style=args.style)
        fig = plotter.plot(
            x_column=args.x_column,
            y_columns=args.y_columns,
            group_by=args.group_by,
            output_path=output_path,
            title=title,
            x_label=args.x_label,
            y_labels=args.y_labels,
            x_padding=args.x_padding,
            y_padding=args.y_padding,
            figsize=tuple(args.figsize),
            legend_loc=args.legend_loc,
            legend_outside=args.legend_outside,
            legend_bbox=tuple(args.legend_bbox),
            marker=args.marker,
            markersize=args.markersize,
            linewidth=args.linewidth,
            subplots=args.subplots,
            show_grid=not args.no_grid,
            dpi=args.dpi,
        )

    # 如果没有指定输出路径，尝试显示图片
    if not args.output:
        try:
            plt.show()
        except Exception:
            print("[提示] 无法显示图片。请使用 --output 保存到文件。")


if __name__ == "__main__":
    main()
