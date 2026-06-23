# encoding: utf8
"""
使用 statsmodels 构建 Logistic Regression，分析相似度分数与二分类标签之间的关系。

功能：
  1. 从 CSV 加载逐层相似度矩阵，结合数据集标签构建逻辑回归模型
  2. 使用 statsmodels 进行逻辑回归拟合
  3. 报告全部统计指标：
     - 系数表：coef, std err, z, p-value, 95% CI
     - 优势比 (odds ratio) 及 CI
     - 边际效应 (marginal effect)
     - 伪 R²: McFadden's, McFadden's adjusted, Cox-Snell, Nagelkerke
     - 对数似然值 (log-likelihood)
     - AIC / BIC
     - 似然比检验 (LR test)
     - 分类准确率、精确率、召回率、特异度、F1
     - ROC AUC
     - Hosmer-Lemeshow 拟合优度检验

用法：
  # 命令行：加载相似度 CSV 和数据集
  python logistic_regression.py \
      --similarities output/similarities_Qwen_mean.csv \
      --data_file spatial_info_annotation/xxx.json \
      --output_dir ./logistic_results \
      --model_alias Qwen9B_mean

  # 作为模块使用
  from logistic_regression import run_logistic_regression
  results = run_logistic_regression(similarities, labels)
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import read_data

# 抑制 statsmodels 的 ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Hosmer-Lemeshow 拟合优度检验
# =============================================================================


def hosmer_lemeshow_test(y, y_pred_proba, groups=10):
    """Hosmer-Lemeshow 拟合优度检验。

    将样本按预测概率排序后等分为 groups 组，
    比较各组的观测事件数与期望事件数，计算卡方统计量。

    原假设：模型拟合良好。
    若 p < 0.05 则拒绝原假设，表明模型拟合不佳。

    Args:
        y:            实际标签 (0/1)
        y_pred_proba: 预测概率
        groups:       分组数（默认 10）

    Returns:
        dict: {chisq, p_value, df}
    """
    n = len(y)
    order = np.argsort(y_pred_proba)
    y_sorted = np.array(y)[order]
    proba_sorted = np.array(y_pred_proba)[order]

    group_size = n // groups
    remainder = n % groups

    chisq = 0.0
    actual_freedom = 0

    start = 0
    for g in range(groups):
        size = group_size + (1 if g < remainder else 0)
        if size == 0:
            continue

        end = start + size
        group_y = y_sorted[start:end]
        group_proba = proba_sorted[start:end]

        observed_1 = group_y.sum()
        observed_0 = size - observed_1
        expected_1 = group_proba.sum()
        expected_0 = size - expected_1

        # 避免除零
        if expected_1 > 0 and expected_0 > 0:
            chisq += (observed_1 - expected_1) ** 2 / expected_1
            chisq += (observed_0 - expected_0) ** 2 / expected_0
            actual_freedom += 1
        elif expected_1 > 0:
            chisq += (observed_1 - expected_1) ** 2 / expected_1
            actual_freedom += 1
        elif expected_0 > 0:
            chisq += (observed_0 - expected_0) ** 2 / expected_0
            actual_freedom += 1

        start = end

    df = max(1, actual_freedom - 2)  # 自由度 = 有效组数 - 2
    p_value = 1.0 - scipy_stats.chi2.cdf(chisq, df)

    return {"chisq": chisq, "p_value": p_value, "df": df}


# =============================================================================
# 核心：单变量逻辑回归
# =============================================================================


def run_logistic_regression(similarities, labels, label_name="similarity"):
    """对一组相似度分数和二分类标签执行完整的逻辑回归分析。

    y = 1 / (1 + exp(-(β₀ + β₁·x)))

    Args:
        similarities: 相似度序列 (list/array, shape=(n,))
        labels:       二分类标签 (list/array, shape=(n,)), 0/1
        label_name:   自变量名称（用于输出显示）

    Returns:
        dict 或 None: 所有统计指标；拟合失败时返回 None
    """
    # ---------- 数据准备 ----------
    X = np.asarray(similarities, dtype=float)
    y = np.asarray(labels, dtype=int)

    # 去除 NaN / Inf
    valid = ~(np.isnan(X) | np.isinf(X) | np.isnan(y))
    X = X[valid]
    y = y[valid]

    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("没有有效样本（全部为 NaN / Inf）。")

    n_positive = int(y.sum())
    n_negative = n_samples - n_positive

    # 检查是否全为同一类别
    if n_positive == 0 or n_negative == 0:
        print(
            f"  [警告] 标签全为 {'正例' if n_positive > 0 else '负例'}，"
            f"无法拟合逻辑回归。"
        )
        return None

    # 添加截距项
    X_with_const = sm.add_constant(X, has_constant="add")

    # ---------- 模型拟合 ----------
    model = sm.Logit(y, X_with_const)

    # 尝试多种优化方法
    for method in ["newton", "bfgs", "lbfgs", "nm"]:
        try:
            result = model.fit(disp=False, method=method, maxiter=2000)
            break
        except Exception:
            continue
    else:
        print(f"  [错误] 所有优化方法均未能收敛。")
        return None

    if not result.mle_retvals.get("converged", True):
        print(f"  [警告] 优化可能未完全收敛，请检查结果。")

    # ---------- 提取统计指标 ----------

    # --- 系数表 ---
    param_names = ["const", label_name]
    summary_table = []

    for i, name in enumerate(param_names):
        # 兼容 Series / ndarray
        coef = float(result.params.iloc[i]) if hasattr(result.params, "iloc") else float(result.params[i])
        se = float(result.bse.iloc[i]) if hasattr(result.bse, "iloc") else float(result.bse[i])
        z_val = float(result.tvalues.iloc[i]) if hasattr(result.tvalues, "iloc") else float(result.tvalues[i])
        p_val = float(result.pvalues.iloc[i]) if hasattr(result.pvalues, "iloc") else float(result.pvalues[i])

        ci = result.conf_int()
        ci_lower = float(ci.iloc[i, 0]) if hasattr(ci, "iloc") else float(ci[i, 0])
        ci_upper = float(ci.iloc[i, 1]) if hasattr(ci, "iloc") else float(ci[i, 1])

        # 优势比
        odds_ratio = np.exp(coef)
        odds_ratio_ci_lower = np.exp(ci_lower)
        odds_ratio_ci_upper = np.exp(ci_upper)

        # 边际效应（仅对自变量，非截距）
        if i == 1:
            avg_prob = float(np.mean(result.predict()))
            marginal_effect = coef * avg_prob * (1.0 - avg_prob)
        else:
            marginal_effect = None

        summary_table.append({
            "term": name,
            "coef": coef,
            "std_err": se,
            "z": z_val,
            "p_value": p_val,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "odds_ratio": odds_ratio,
            "odds_ratio_ci_95_lower": odds_ratio_ci_lower,
            "odds_ratio_ci_95_upper": odds_ratio_ci_upper,
            "marginal_effect": marginal_effect,
        })

    # --- 模型拟合优度 ---
    ll_null = float(result.llnull)   # 仅截距模型
    ll_model = float(result.llf)     # 完整模型

    # McFadden's pseudo R²
    mcfadden_r2 = 1.0 - ll_model / ll_null if ll_null != 0.0 else np.nan

    # McFadden's adjusted R²
    k = 1  # 自变量个数（不含截距）
    mcfadden_r2_adj = 1.0 - (ll_model - k) / ll_null if ll_null != 0.0 else np.nan

    # Cox-Snell pseudo R²
    cox_snell_r2 = 1.0 - np.exp(-2.0 * (ll_model - ll_null) / n_samples)

    # Nagelkerke pseudo R² (max(Cox-Snell) = 1 - exp(2*ll_null/n))
    max_cs = 1.0 - np.exp(2.0 * ll_null / n_samples)
    nagelkerke_r2 = cox_snell_r2 / max_cs if max_cs != 0.0 else np.nan

    # --- 似然比检验 ---
    lr_stat = 2.0 * (ll_model - ll_null)
    lr_df = k
    lr_pvalue = 1.0 - scipy_stats.chi2.cdf(lr_stat, lr_df)

    # --- 信息准则 ---
    aic = float(result.aic)
    bic = float(result.bic)

    # --- 预测与分类指标 ---
    y_pred_proba = np.asarray(result.predict())
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = float(accuracy_score(y, y_pred))
    precision = float(precision_score(y, y_pred, zero_division=0))
    recall = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    specificity = float(recall_score(1 - y, 1 - y_pred, zero_division=0))

    # ROC AUC
    try:
        roc_auc = float(roc_auc_score(y, y_pred_proba))
    except ValueError:
        roc_auc = np.nan

    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    else:
        tn = fp = fn = tp = 0

    # --- Hosmer-Lemeshow 检验 ---
    hl_test = hosmer_lemeshow_test(y, y_pred_proba)

    # --- 组装结果 ---
    return {
        # 基本描述
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_ratio": float(n_positive / n_samples),

        # 系数表
        "coefficients": summary_table,

        # 模型拟合统计
        "log_likelihood_null": ll_null,
        "log_likelihood_model": ll_model,
        "mcfadden_r2": float(mcfadden_r2),
        "mcfadden_r2_adjusted": float(mcfadden_r2_adj),
        "cox_snell_r2": float(cox_snell_r2),
        "nagelkerke_r2": float(nagelkerke_r2),

        # 似然比检验
        "lr_stat": float(lr_stat),
        "lr_df": int(lr_df),
        "lr_pvalue": float(lr_pvalue),

        # 信息准则
        "aic": aic,
        "bic": bic,

        # 分类指标
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": float(roc_auc),

        # 混淆矩阵
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},

        # 拟合优度检验
        "hosmer_lemeshow_chisq": float(hl_test["chisq"]),
        "hosmer_lemeshow_pvalue": float(hl_test["p_value"]),
        "hosmer_lemeshow_df": int(hl_test["df"]),

        # statsmodels 完整 summary 字符串
        "summary": str(result.summary()),
    }


# =============================================================================
# 格式化打印
# =============================================================================


def print_results(results):
    """格式化打印逻辑回归的全部统计结果。"""
    if results is None:
        print("逻辑回归未能拟合。")
        return

    print("=" * 72)
    print("  Logistic Regression 分析结果 (statsmodels)")
    print("=" * 72)

    # --- 数据概况 ---
    print(f"\n  ── 数据概况 ──")
    print(f"  有效样本数:     {results['n_samples']}")
    print(f"  正例 (1):       {results['n_positive']}")
    print(f"  负例 (0):       {results['n_negative']}")
    print(f"  正例比例:       {results['positive_ratio']:.4f}")

    # --- 系数估计 ---
    print(f"\n  ── 系数估计 ──")
    header = (
        f"  {'Term':<14} {'Coef':>10} {'Std.Err':>10} "
        f"{'z':>8} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results["coefficients"]:
        print(
            f"  {row['term']:<14} {row['coef']:>10.5f} {row['std_err']:>10.5f} "
            f"{row['z']:>8.3f} {row['p_value']:>10.5f} "
            f"{row['ci_95_lower']:>10.5f} {row['ci_95_upper']:>10.5f}"
        )

    # --- 优势比 ---
    print(f"\n  ── 优势比 (Odds Ratios) ──")
    print(f"  {'Term':<14} {'OR':>10} {'[0.025':>10} {'0.975]':>10}")
    print("  " + "-" * 46)
    for row in results["coefficients"]:
        print(
            f"  {row['term']:<14} {row['odds_ratio']:>10.4f} "
            f"{row['odds_ratio_ci_95_lower']:>10.4f} "
            f"{row['odds_ratio_ci_95_upper']:>10.4f}"
        )

    # --- 边际效应 ---
    for row in results["coefficients"]:
        if row["marginal_effect"] is not None:
            me = row["marginal_effect"]
            print(
                f"\n  边际效应 ({row['term']}): {me:.5f}  "
                f"(similarity 增加 0.01, 概率平均变化约 {me * 0.01:.5f})"
            )

    # --- 模型拟合优度 ---
    print(f"\n  ── 模型拟合优度 ──")
    print(f"  Log-Likelihood (null):      {results['log_likelihood_null']:>12.4f}")
    print(f"  Log-Likelihood (model):     {results['log_likelihood_model']:>12.4f}")
    print(f"  McFadden's R²:              {results['mcfadden_r2']:>12.5f}")
    print(f"  McFadden's Adjusted R²:     {results['mcfadden_r2_adjusted']:>12.5f}")
    print(f"  Cox-Snell R²:               {results['cox_snell_r2']:>12.5f}")
    print(f"  Nagelkerke R²:              {results['nagelkerke_r2']:>12.5f}")

    # --- 似然比检验 ---
    print(f"\n  ── 似然比检验 (Likelihood Ratio Test) ──")
    print(
        f"  χ² = {results['lr_stat']:.4f},  "
        f"df = {results['lr_df']},  "
        f"p = {results['lr_pvalue']:.6f}"
    )
    if results["lr_pvalue"] < 0.05:
        print(f"  → 模型显著优于仅截距模型 (p < 0.05)")
    else:
        print(f"  → 模型未显著优于仅截距模型 (p ≥ 0.05)")

    # --- 信息准则 ---
    print(f"\n  ── 信息准则 ──")
    print(f"  AIC: {results['aic']:>12.4f}")
    print(f"  BIC: {results['bic']:>12.4f}")

    # --- 分类性能 ---
    print(f"\n  ── 分类性能 (threshold = 0.5) ──")
    print(f"  准确率  (Accuracy):    {results['accuracy']:.5f}")
    print(f"  精确率  (Precision):   {results['precision']:.5f}")
    print(f"  召回率  (Recall):      {results['recall']:.5f}")
    print(f"  特异度  (Specificity): {results['specificity']:.5f}")
    print(f"  F1 分数:               {results['f1']:.5f}")
    print(f"  ROC AUC:               {results['roc_auc']:.5f}")

    # --- 混淆矩阵 ---
    cm = results["confusion_matrix"]
    print(f"\n  混淆矩阵:")
    print(f"               预测 0    预测 1")
    print(f"  实际 0       {cm['tn']:>5}     {cm['fp']:>5}")
    print(f"  实际 1       {cm['fn']:>5}     {cm['tp']:>5}")

    # --- Hosmer-Lemeshow 检验 ---
    print(f"\n  ── Hosmer-Lemeshow 拟合优度检验 ──")
    hl_chi = results["hosmer_lemeshow_chisq"]
    hl_p = results["hosmer_lemeshow_pvalue"]
    hl_df = results["hosmer_lemeshow_df"]
    print(f"  χ² = {hl_chi:.4f},  df = {hl_df},  p = {hl_p:.5f}")
    if hl_p < 0.05:
        print(f"  → 模型拟合不佳 (p < 0.05, 观测值与预测值差异显著)")
    else:
        print(f"  → 模型拟合良好 (p ≥ 0.05)")

    # --- statsmodels 完整 summary ---
    print(f"\n  " + "─" * 70)
    print(f"  statsmodels 完整输出:")
    print(f"  " + "─" * 70)
    for line in results["summary"].splitlines():
        print(f"  {line}")

    print(f"\n" + "=" * 72)


# =============================================================================
# 工具函数：结果导出
# =============================================================================


def results_to_dataframe(results):
    """将单次回归结果转为一行 DataFrame，便于拼接和保存。"""
    if results is None:
        return pd.DataFrame()

    rows = [
        ("n_samples", results["n_samples"]),
        ("n_positive", results["n_positive"]),
        ("n_negative", results["n_negative"]),
        ("positive_ratio", results["positive_ratio"]),
        ("log_likelihood_null", results["log_likelihood_null"]),
        ("log_likelihood_model", results["log_likelihood_model"]),
        ("mcfadden_r2", results["mcfadden_r2"]),
        ("mcfadden_r2_adjusted", results["mcfadden_r2_adjusted"]),
        ("cox_snell_r2", results["cox_snell_r2"]),
        ("nagelkerke_r2", results["nagelkerke_r2"]),
        ("aic", results["aic"]),
        ("bic", results["bic"]),
        ("lr_stat", results["lr_stat"]),
        ("lr_df", results["lr_df"]),
        ("lr_pvalue", results["lr_pvalue"]),
        ("accuracy", results["accuracy"]),
        ("precision", results["precision"]),
        ("recall", results["recall"]),
        ("specificity", results["specificity"]),
        ("f1", results["f1"]),
        ("roc_auc", results["roc_auc"]),
        ("tn", results["confusion_matrix"]["tn"]),
        ("fp", results["confusion_matrix"]["fp"]),
        ("fn", results["confusion_matrix"]["fn"]),
        ("tp", results["confusion_matrix"]["tp"]),
        ("hosmer_lemeshow_chisq", results["hosmer_lemeshow_chisq"]),
        ("hosmer_lemeshow_pvalue", results["hosmer_lemeshow_pvalue"]),
        ("hosmer_lemeshow_df", results["hosmer_lemeshow_df"]),
    ]

    # 添加系数
    for coeff in results["coefficients"]:
        term = coeff["term"]
        rows.append((f"{term}_coef", coeff["coef"]))
        rows.append((f"{term}_std_err", coeff["std_err"]))
        rows.append((f"{term}_z", coeff["z"]))
        rows.append((f"{term}_p_value", coeff["p_value"]))
        rows.append((f"{term}_odds_ratio", coeff["odds_ratio"]))
        if coeff["marginal_effect"] is not None:
            rows.append((f"{term}_marginal_effect", coeff["marginal_effect"]))

    return pd.DataFrame(rows, columns=["metric", "value"])


# =============================================================================
# 命令行入口
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="使用 statsmodels 构建 Logistic Regression 分析相似度与标签的关系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个相似度 CSV 的全部层
  python logistic_regression.py \\
      --similarities output/similarities_Qwen_prompt1_mean.csv \\
      --data_file spatial_info_annotation/xxx.json

  # 指定输出和层范围
  python logistic_regression.py \\
      --similarities output/similarities.csv \\
      --data_file data.json \\
      --output_dir ./logistic_results \\
      --model_alias Qwen9B_p1_mean \\
      --layers 0 1 2 3 4

  # 跨层聚合分析
  python logistic_regression.py \\
      --similarities output/similarities.csv \\
      --data_file data.json \\
      --aggregate mean
        """,
    )

    parser.add_argument(
        "--similarities", "-s", type=str, required=True,
        help="相似度 CSV 文件路径（行=句子对, 列=层）",
    )
    parser.add_argument(
        "--data_file", "-d", type=str, required=True,
        help="数据集 JSON 文件路径（用于获取标签）",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./logistic_output",
        help="输出目录（默认: ./logistic_output）",
    )
    parser.add_argument(
        "--model_alias", "-a", type=str, default="model",
        help="模型别名（用于输出文件命名）",
    )
    parser.add_argument(
        "--layers", "-l", type=int, nargs="*", default=None,
        help="指定分析的层索引（默认: 全部层）。例如: -l 0 5 10",
    )
    parser.add_argument(
        "--aggregate", type=str, default=None,
        choices=["mean", "max", "min"],
        help="跨层聚合方式（不指定则逐层分析）",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------

    # 1. 相似度矩阵
    print(f"[加载] 相似度矩阵: {args.similarities}")
    sim_df = pd.read_csv(args.similarities)
    n_pairs, n_layers = sim_df.shape
    print(f"  → 形状: {n_pairs} 个句子对 × {n_layers} 层")

    # 2. 标签
    print(f"[加载] 数据集: {args.data_file}")
    dataset = read_data.SpatialDataset(args.data_file)
    labels = np.array(dataset.labels, dtype=int)
    print(f"  → 标签数: {len(labels)}")

    # 3. 长度校验
    if len(labels) != n_pairs:
        print(
            f"[错误] 标签数量 ({len(labels)}) 与相似度行数 ({n_pairs}) 不一致！\n"
            f"  请确认相似度 CSV 文件与数据集文件匹配。"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 确定分析范围
    # ------------------------------------------------------------------
    layer_indices = args.layers if args.layers is not None else list(range(n_layers))

    # ------------------------------------------------------------------
    # 根据相似度文件名称匹配变项，自动创建子目录
    # ------------------------------------------------------------------
    sim_basename = os.path.basename(args.similarities)
    match = re.match(r"similarities_(.*)\.csv", sim_basename)
    if match:
        variant = match.group(1)
        args.output_dir = os.path.join(args.output_dir, variant)
        print(f"[信息] 匹配到变项 \"{variant}\"，输出目录调整为: {args.output_dir}")

    # ------------------------------------------------------------------
    # 创建输出目录
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 执行分析
    # ------------------------------------------------------------------
    all_results = []

    if args.aggregate:
        # —— 跨层聚合 ——
        print(f"\n[分析] 跨层聚合方式: {args.aggregate}")

        if args.aggregate == "mean":
            aggregated = sim_df.iloc[:, layer_indices].mean(axis=1).values
        elif args.aggregate == "max":
            aggregated = sim_df.iloc[:, layer_indices].max(axis=1).values
        elif args.aggregate == "min":
            aggregated = sim_df.iloc[:, layer_indices].min(axis=1).values

        label_name = f"similarity_{args.aggregate}"
        results = run_logistic_regression(aggregated, labels, label_name=label_name)

        if results:
            results["layer"] = args.aggregate
            print_results(results)
            all_results.append(results)
    else:
        # —— 逐层分析 ——
        for layer_idx in layer_indices:
            if layer_idx >= n_layers:
                print(f"[跳过] Layer {layer_idx} 超出范围（最大 {n_layers - 1}）")
                continue

            print(f"\n{'#' * 72}")
            print(f"#  Layer {layer_idx}")
            print(f"{'#' * 72}")

            similarities = sim_df.iloc[:, layer_idx].values
            label_name = f"similarity_L{layer_idx}"

            results = run_logistic_regression(similarities, labels, label_name=label_name)

            if results:
                results["layer"] = layer_idx
                print_results(results)
                all_results.append(results)

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    if not all_results:
        print("\n[警告] 没有成功拟合的回归结果可保存。")
        return

    prefix = f"{args.model_alias}_aggregated" if args.aggregate else args.model_alias

    # --- 汇总表（每层/聚合一行） ---
    summary_rows = []
    for r in all_results:
        row = {
            "layer": r["layer"],
            "n_samples": r["n_samples"],
            "mcfadden_r2": r["mcfadden_r2"],
            "nagelkerke_r2": r["nagelkerke_r2"],
            "aic": r["aic"],
            "bic": r["bic"],
            "log_likelihood": r["log_likelihood_model"],
            "lr_stat": r["lr_stat"],
            "lr_pvalue": r["lr_pvalue"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
            "hl_pvalue": r["hosmer_lemeshow_pvalue"],
        }
        for coeff in r["coefficients"]:
            term = coeff["term"]
            row[f"{term}_coef"] = coeff["coef"]
            row[f"{term}_std_err"] = coeff["std_err"]
            row[f"{term}_p_value"] = coeff["p_value"]
            row[f"{term}_odds_ratio"] = coeff["odds_ratio"]

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    summary_path = os.path.join(args.output_dir, f"logistic_summary_{prefix}.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 汇总表 → {summary_path}")

    # --- 逐层详情 ---
    for r in all_results:
        layer_label = str(r["layer"])
        df_detail = results_to_dataframe(r)
        detail_path = os.path.join(
            args.output_dir, f"logistic_detail_{prefix}_L{layer_label}.csv"
        )
        df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"[保存] 详情 → {detail_path}")

    # --- 完整文本报告 ---
    report_path = os.path.join(args.output_dir, f"logistic_report_{prefix}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Logistic Regression 完整报告\n")
        f.write(f"模型: {args.model_alias}\n")
        f.write(f"数据: {args.data_file}\n")
        f.write(f"相似度文件: {args.similarities}\n")
        f.write(f"层: {layer_indices}\n")
        f.write(f"聚合: {args.aggregate or '逐层'}\n")
        f.write(f"分析时间: {pd.Timestamp.now().isoformat()}\n")
        f.write("=" * 72 + "\n")

        for r in all_results:
            f.write(f"\n{'#' * 72}\n")
            f.write(f"#  Layer: {r['layer']}\n")
            f.write(f"{'#' * 72}\n")
            f.write(r["summary"])
            f.write("\n\n")

    print(f"[保存] 完整报告 → {report_path}")
    print("\n[完成] 逻辑回归分析结束。")


# =============================================================================
# 直接调用入口
# =============================================================================

if __name__ == "__main__":
    main()
