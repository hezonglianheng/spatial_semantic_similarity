# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Optional, Union, Literal, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

PoolingStrategy = Literal["mean", "last_token", "eos_token", "max", "weighted_mean"]
"""支持的池化策略类型"""

class HiddenInfoExtractor:
    def __init__(self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        self.model_path = model_path
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        self.device = device
        print(f"Loading model from {model_path} on {device}")
        self.torch_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        # 对于未设置 pad_token 的 tokenizer（如 LLaMA），将 eos_token 用作 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        # device_map 与 .to() 互斥：若用户通过 kwargs 传入了 device_map，则由
        # accelerate 管理设备分配；否则使用 .to(device) 进行简单的单设备加载。
        if "device_map" not in kwargs:
            self.model.to(device)
        self.model.eval()

    def tokenize(
        self,
        text: Union[str, list[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        """分词函数，支持单句和批量输入。

        Args:
            text:            单个句子字符串或句子列表
            padding:         是否填充到批次内最长长度
            truncation:      是否截断超过模型最大长度的序列
            max_length:      最大序列长度（None 则使用 tokenizer 默认值）
            return_tensors:  返回的张量类型（默认 "pt"）

        Returns:
            dict 包含 input_ids, attention_mask 等
        """
        if isinstance(text, str):
            text = [text]

        tokenizer_kwargs = {
            "return_tensors": return_tensors,
            "padding": padding,
            "truncation": truncation,
        }
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length

        inputs = self.tokenizer(text, **tokenizer_kwargs)
        return inputs

    @torch.no_grad()
    def encode(self, inputs: dict[str, torch.Tensor], output_attentions: bool = False):
        """将输入编码为LM隐状态

        Args:
            inputs (dict[str, torch.Tensor]): Transformers分词器标准输入
            output_attentions (bool): 是否同时输出注意力权重（weighted_mean池化需要）

        Returns:
            Transformers模型标注输出，包含hidden_state和attention_weights信息
        """
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=output_attentions)
        return outputs

    def pipeline(self, text: str):
        """将文本输入通过模型处理，返回隐状态和注意力权重信息

        Args:
            text (str): 输入文本

        Returns:
            Transformers模型标注输出，包含hidden_state和attention_weights信息
        """
        inputs = self.tokenize(text)
        outputs = self.encode(inputs)
        return outputs

    @torch.no_grad()
    def get_layer_hidden_state(
        self,
        inputs: dict[str, torch.Tensor],
        layer_index: int,
    ) -> torch.Tensor:
        """获取指定层的 hidden state。

        Args:
            inputs: transformers 分词器的标准输出字典，
                    需包含 input_ids 和 attention_mask。
            layer_index: 层索引（从 0 开始）。
                         0 表示 embedding 层输出，
                         1 表示第一层 transformer 层输出，依此类推。
                         负索引（如 -1）表示从最后一层倒数。

        Returns:
            torch.Tensor: shape = (batch_size, seq_len, hidden_dim)
                          指定层的 hidden state。

        Raises:
            IndexError: 当 layer_index 超出有效范围时抛出。

        Example:
            >>> extractor = HiddenInfoExtractor("path/to/model")
            >>> inputs = extractor.tokenize("你好，世界")
            >>> # 获取最后一层的 hidden state
            >>> last_hidden = extractor.get_layer_hidden_state(inputs, layer_index=-1)
            >>> # 获取第 12 层的 hidden state
            >>> layer_12 = extractor.get_layer_hidden_state(inputs, layer_index=12)
        """
        outputs = self.encode(inputs)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

        num_layers = len(hidden_states)

        # 处理负索引
        resolved_index = layer_index
        if resolved_index < 0:
            resolved_index = num_layers + resolved_index

        if resolved_index < 0 or resolved_index >= num_layers:
            raise IndexError(
                f"层索引 {layer_index}（解析为 {resolved_index}）超出范围 "
                f"[0, {num_layers - 1}]，共 {num_layers} 层"
            )

        return hidden_states[resolved_index]

    # ==================================================================
    # 池化策略注册表
    # ==================================================================

    _POOLING_REGISTRY: dict[str, Any] = {}
    """池化策略注册表：名称 → 可调用对象。

    向注册表添加新策略的方式：

        HiddenInfoExtractor._POOLING_REGISTRY["my_strategy"] = my_pool_fn

    策略函数签名要求：

        def my_pool(
            hidden_states: torch.Tensor,      # (batch, seq_len, hidden_dim)
            attention_mask: torch.Tensor,     # (batch, seq_len)
            *,
            attentions: torch.Tensor | None = None,
            input_ids: torch.Tensor | None = None,
            eos_token_id: int | None = None,
        ) -> torch.Tensor: ...
    """

    # ==================================================================
    # 池化策略实现
    # ==================================================================

    @staticmethod
    def _pool_mean(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        attentions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """均值池化：对所有非填充 token 取平均。"""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask_expanded
        summed = masked.sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        return summed / counts

    @staticmethod
    def _pool_last_token(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        attentions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """末位池化：取每个序列最后一个非填充 token 的隐藏状态。"""
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lengths, :]

    @staticmethod
    def _pool_eos_token(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        attentions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """EOS 池化：取 EOS token 位置的隐藏状态。

        通过 input_ids 与 eos_token_id 精确匹配来定位 EOS token，
        而非简单取最后一个非填充 token。
        若某序列中未找到 eos_token，则回退到末位 token。
        """
        if input_ids is None:
            raise ValueError(
                "eos_token 策略需要 input_ids 参数。"
                "请确保调用方传入了 input_ids 张量。"
            )
        if eos_token_id is None:
            raise ValueError(
                "eos_token 策略需要 eos_token_id 参数。"
                "请从 tokenizer.eos_token_id 获取后传入。"
            )
        # 找到每个序列中 eos_token 的位置（取第一个匹配）
        eos_mask = (input_ids == eos_token_id) & attention_mask.bool()
        has_eos = eos_mask.any(dim=1)                       # (batch,)
        eos_positions = eos_mask.float().argmax(dim=1)      # (batch,) —— 无 True 时 argmax 返回 0

        # 对没有 eos_token 的序列，回退到最后一个非填充 token
        if not has_eos.all():
            fallback_positions = attention_mask.sum(dim=1) - 1
            eos_positions = torch.where(has_eos, eos_positions, fallback_positions)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, eos_positions, :]

    @staticmethod
    def _pool_max(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        attentions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """最大池化：对每个维度取非填充区域的最大值。"""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask_expanded + (1 - mask_expanded) * -1e9
        return masked.max(dim=1).values

    @staticmethod
    def _pool_weighted_mean(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        attentions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """注意力加权均值池化：用最后一层注意力权重对 token 做加权平均。"""
        if not attentions:
            raise ValueError(
                "weighted_mean 策略需要传入注意力权重，但 attentions 为空。"
                "可能原因：\n"
                "1. 未设置 output_attentions=True；\n"
                "2. 模型使用了不支持返回注意力权重的后端（如 Flash Attention 2 / SDPA）。"
                "请尝试在加载模型时显式指定 attn_implementation='eager'，"
                "例如: AutoModelForCausalLM.from_pretrained(..., attn_implementation='eager')"
            )
        last_attn = attentions[-1]                          # (batch, num_heads, seq_len, seq_len)
        avg_attn = last_attn.mean(dim=1)                    # (batch, seq_len, seq_len)
        weights = avg_attn.sum(dim=-1)                      # (batch, seq_len)
        weights = weights * attention_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)

    # ==================================================================
    # 池化分派
    # ==================================================================

    @staticmethod
    def _pool_single_layer(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy: str = "mean",
        attentions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """对单层 token-level 隐藏状态执行池化，得到句向量。

        通过注册表 _POOLING_REGISTRY 分派到对应的池化函数。

        Args:
            hidden_states:  shape = (batch_size, seq_len, hidden_dim)
            attention_mask: shape = (batch_size, seq_len) —— 1 表示真实 token，0 表示 padding
            strategy:       池化策略名称（需已在 _POOLING_REGISTRY 中注册）
            attentions:     注意力权重 (batch, num_heads, seq_len, seq_len)，仅 weighted_mean 需要
            input_ids:      token ID 序列 (batch, seq_len)，eos_token 策略需要
            eos_token_id:   EOS token 的 ID，eos_token 策略需要

        Returns:
            句向量 (batch_size, hidden_dim)
        """
        pool_fn = HiddenInfoExtractor._POOLING_REGISTRY.get(strategy)
        if pool_fn is None:
            available = list(HiddenInfoExtractor._POOLING_REGISTRY.keys())
            raise ValueError(
                f"不支持的池化策略: {strategy}。"
                f"可选: {', '.join(available)}"
            )

        # ── 设备对齐：确保所有张量与 hidden_states 在同一设备 ──
        target_device = hidden_states.device
        if attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        if input_ids is not None and input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if attentions is not None:
            attentions = tuple(
                a.to(target_device) if a.device != target_device else a
                for a in attentions
            )

        sent_vec = pool_fn(
            hidden_states,
            attention_mask,
            attentions=attentions,
            input_ids=input_ids,
            eos_token_id=eos_token_id,
        )
        # 池化结果立即移至 CPU，释放 GPU 显存；后续拼接/归一化/相似度均在 CPU 完成
        return sent_vec.cpu()

    # ==================================================================
    # 句嵌入提取
    # ==================================================================

    @staticmethod
    def get_token_embedding_from_slice(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        slice_spec: Union[slice, list[int], tuple[int, int], tuple[int, int, int]],
        *,
        input_ids: Optional[torch.Tensor] = None,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """根据切片值，从 hidden_states 中提取指定位置 token 的嵌入向量并取算术平均。

        支持多种切片规格形式：
        - Python slice 对象: slice(2, 5)、slice(0, -1)
        - 元组 (start, end): (2, 5) 或 (start, end, step): (0, 10, 2)
        - 显式索引列表: [0, 3, 5, 7]

        仅对有效 token（attention_mask=1）的嵌入做平均，padding token 会被自动排除。
        若切片范围内无有效 token，则返回零向量。

        Args:
            hidden_states:  shape = (batch_size, seq_len, hidden_dim) 或 (seq_len, hidden_dim)
            attention_mask: shape = (batch_size, seq_len) 或 (seq_len,) —— 1=有效token, 0=padding
            slice_spec:     切片规格，支持 slice / tuple(start, end[, step]) / 索引列表
            input_ids:      可选，token ID 张量 (batch, seq_len)，仅供调用方调试校验
            return_numpy:   返回 numpy 数组（True）还是 torch Tensor（False）

        Returns:
            切片范围内有效 token 嵌入向量的算术均值:
            - 输入为 batch: (batch_size, hidden_dim)
            - 输入为单句:   (hidden_dim,)

        Example:
            >>> extractor = HiddenInfoExtractor("path/to/model")
            >>> inputs = extractor.tokenize("一只猫坐在垫子上")
            >>> outputs = extractor.encode(inputs)
            >>> # 取最后一层 hidden state
            >>> hs = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            >>> # 取第 2~5 个 token 的嵌入均值（Python slice 风格）
            >>> vec = HiddenInfoExtractor.get_token_embedding_from_slice(
            ...     hs, inputs["attention_mask"], slice(2, 5)
            ... )
            >>> # 用元组方式等价表达
            >>> vec = HiddenInfoExtractor.get_token_embedding_from_slice(
            ...     hs, inputs["attention_mask"], (2, 5)
            ... )
            >>> # 取指定位置列表
            >>> vec = HiddenInfoExtractor.get_token_embedding_from_slice(
            ...     hs, inputs["attention_mask"], [0, 2, 4]
            ... )
        """
        # ── 统一为 3D: (batch_size, seq_len, hidden_dim) ──
        squeeze_output = False
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # ── 将切片规格解析为位置索引列表 ──
        if isinstance(slice_spec, slice):
            indices = list(range(seq_len))[slice_spec]
        elif isinstance(slice_spec, tuple):
            indices = list(range(seq_len))[slice(*slice_spec)]
        elif isinstance(slice_spec, list):
            indices = slice_spec
        else:
            raise TypeError(
                f"slice_spec 类型不支持: {type(slice_spec)}。"
                f"请使用 slice、tuple(start, end[, step]) 或索引列表。"
            )

        if len(indices) == 0:
            # 切片为空列表 → 返回零向量
            target_device = hidden_states.device
            result = torch.zeros(batch_size, hidden_dim, device=target_device)
            if squeeze_output:
                result = result.squeeze(0)
            result = result.cpu()
            if return_numpy:
                return result.float().numpy()
            return result

        # 构建切片位置掩码 (batch_size, seq_len)
        slice_mask = torch.zeros(batch_size, seq_len, device=hidden_states.device)
        slice_mask[:, indices] = 1.0

        # 与 attention_mask 取交集：只保留切片内且非 padding 的 token
        attn_mask = attention_mask.float().to(hidden_states.device)
        effective_mask = slice_mask * attn_mask       # (batch_size, seq_len)

        # ── 算术平均 ──
        mask_expanded = effective_mask.unsqueeze(-1)   # (batch, seq_len, 1)
        masked = hidden_states * mask_expanded
        summed = masked.sum(dim=1)                      # (batch, hidden_dim)
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)，避免除以零

        result = summed / counts

        if squeeze_output:
            result = result.squeeze(0)

        result = result.cpu()
        if return_numpy:
            return result.float().numpy()
        return result

    @staticmethod
    def extract_sentence_embedding(
        hidden_states: tuple,
        attention_mask: torch.Tensor,
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        attentions: Optional[tuple] = None,
        input_ids: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """从模型输出的 hidden_states 中提取句嵌入向量。

        对每一层执行池化，将 (batch, seq_len, hidden_dim) 聚合为 (batch, hidden_dim)，
        最终堆叠为 (num_layers, batch, hidden_dim)。

        Args:
            hidden_states:  tuple of (batch, seq_len, hidden_dim)，每层一个 tensor
            attention_mask: (batch, seq_len) —— 1 表示真实 token，0 表示 padding
            pooling:        池化策略，可选: mean, last_token, eos_token, max, weighted_mean
            layers:         指定要返回的层索引（None = 全部层）。0 为 embedding 层，-1 为最后一层
            attentions:     attention weights tuple，仅 weighted_mean 策略需要
            input_ids:      token ID 序列 (batch, seq_len)，eos_token 策略需要
            eos_token_id:   EOS token 的 ID，eos_token 策略需要
            normalize:      是否对句向量做 L2 归一化
            return_numpy:   返回 numpy 数组（True）还是 torch Tensor（False）

        Returns:
            embeddings: shape = (num_layers, num_sentences, hidden_dim)

        Example:
            >>> extractor = HiddenInfoExtractor("path/to/model")
            >>> inputs = extractor.tokenize("一只猫坐在垫子上")
            >>> outputs = extractor.encode(inputs)
            >>> # 获取所有层的 mean-pooled 句嵌入
            >>> emb = HiddenInfoExtractor.extract_sentence_embedding(
            ...     outputs.hidden_states, inputs["attention_mask"], pooling="mean"
            ... )
            >>> emb.shape  # (num_layers, 1, hidden_dim)
        """
        num_layers_total = len(hidden_states)

        # 对每层执行池化
        pooled: list[torch.Tensor] = []
        for hs in hidden_states:
            sent_vec = HiddenInfoExtractor._pool_single_layer(
                hs, attention_mask, pooling,
                attentions=attentions,
                input_ids=input_ids,
                eos_token_id=eos_token_id,
            )
            pooled.append(sent_vec)

        # 堆叠为 (num_layers, batch_size, hidden_dim)
        embeddings_tensor = torch.stack(pooled, dim=0)

        # 筛选指定层
        if layers is not None:
            resolved_layers = []
            for l in layers:
                if l < 0:
                    l = num_layers_total + l
                if l < 0 or l >= num_layers_total:
                    raise IndexError(
                        f"层索引 {l} 超出范围 [0, {num_layers_total - 1}]"
                    )
                resolved_layers.append(l)
            embeddings_tensor = embeddings_tensor[resolved_layers, :, :]

        # L2 归一化（防止零向量产生 NaN）
        if normalize:
            embeddings_tensor = torch.nn.functional.normalize(
                embeddings_tensor, p=2, dim=-1
            )
            embeddings_tensor = torch.nan_to_num(embeddings_tensor, nan=0.0)

        if return_numpy:
            return embeddings_tensor.numpy()
        return embeddings_tensor

    # ==================================================================
    # 便捷流水线: tokenize → encode → extract_sentence_embedding
    # ==================================================================

    @torch.no_grad()
    def embed(
        self,
        sentences: Union[str, list[str]],
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        output_attentions: bool = False,
        batch_size: int = 8,
        normalize: bool = True,
        return_numpy: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """完整的句子 → 句嵌入流水线（一步到位）。

        等价于依次调用 tokenize() → encode() → extract_sentence_embedding()，
        并自动处理大批量句子的分批与拼接。

        Args:
            sentences:         单个句子字符串或句子列表
            pooling:           池化策略（默认: mean）
            layers:            指定层索引（None = 全部层）
            output_attentions: 是否输出注意力权重（weighted_mean 策略需要）
            batch_size:        批量处理大小
            normalize:         是否 L2 归一化
            return_numpy:      返回 numpy 数组
            show_progress:     是否显示进度条

        Returns:
            embeddings: shape = (num_layers, num_sentences, hidden_dim)

        Example:
            >>> extractor = HiddenInfoExtractor("path/to/model")
            >>> emb = extractor.embed(["句子1", "句子2"], pooling="mean")
            >>> emb.shape  # (num_layers, 2, hidden_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if len(sentences) == 0:
            raise ValueError("sentences 不能为空列表。请至少提供一个句子。")

        all_pooled_per_batch: list[list[torch.Tensor]] = []

        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="[Embed]", unit="batch")
            except ImportError:
                pass

        for start in iterator:
            batch_sentences = sentences[start:start + batch_size]

            # Step 1: tokenize
            inputs = self.tokenize(batch_sentences)

            # Step 2: encode
            outputs = self.encode(inputs, output_attentions=output_attentions)
            hs_tuple = outputs.hidden_states
            attn_mask = inputs["attention_mask"]
            attns = outputs.attentions if output_attentions else None

            # Step 3: 逐层池化（不在这里做 normalize，最后统一做）
            batch_pooled: list[torch.Tensor] = []
            for hs in hs_tuple:
                sent_vec = self._pool_single_layer(
                    hs, attn_mask, pooling,
                    attentions=attns,
                    input_ids=inputs.get("input_ids"),
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                batch_pooled.append(sent_vec)

            all_pooled_per_batch.append(batch_pooled)

        # 按层拼接所有 batch
        num_layers_total = len(all_pooled_per_batch[0])
        pooled_embeddings: list[torch.Tensor] = []
        for layer_idx in range(num_layers_total):
            layer_emb = torch.cat(
                [batch_emb[layer_idx] for batch_emb in all_pooled_per_batch],
                dim=0,
            )
            pooled_embeddings.append(layer_emb)

        # 堆叠为 (num_layers, num_sentences, hidden_dim)
        embeddings_tensor = torch.stack(pooled_embeddings, dim=0)

        # 筛选指定层
        if layers is not None:
            resolved_layers = []
            for l in layers:
                if l < 0:
                    l = num_layers_total + l
                if l < 0 or l >= num_layers_total:
                    raise IndexError(
                        f"层索引 {l} 超出范围 [0, {num_layers_total - 1}]"
                    )
                resolved_layers.append(l)
            embeddings_tensor = embeddings_tensor[resolved_layers, :, :]

        # L2 归一化（防止零向量产生 NaN）
        if normalize:
            embeddings_tensor = torch.nn.functional.normalize(
                embeddings_tensor, p=2, dim=-1
            )
            embeddings_tensor = torch.nan_to_num(embeddings_tensor, nan=0.0)

        if return_numpy:
            return embeddings_tensor.float().numpy()
        return embeddings_tensor

    # ------------------------------------------------------------------
    # 相似度计算（静态方法：从句嵌入直接计算）
    # ------------------------------------------------------------------

    @staticmethod
    def similarity_from_embeddings(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> dict:
        """从预计算的句嵌入向量计算余弦相似度。

        支持单层和多层嵌入：
        - 单层: shape = (hidden_dim,)，自动扩展为 (1, hidden_dim)
        - 多层: shape = (num_layers, hidden_dim)

        Args:
            embedding_a: 句子 A 的嵌入向量，shape (hidden_dim,) 或 (num_layers, hidden_dim)
            embedding_b: 句子 B 的嵌入向量，shape (hidden_dim,) 或 (num_layers, hidden_dim)

        Returns:
            dict: {
                "layer_similarities": list[float] — 各层余弦相似度
                "avg_similarity": float           — 层间平均相似度
                "max_similarity": float           — 层间最大相似度
                "min_similarity": float           — 层间最小相似度
            }

        Example:
            >>> emb = extractor.embed(["句子1", "句子2"], pooling="mean")
            >>> a, b = emb[:, 0, :], emb[:, 1, :]  # (num_layers, hidden_dim)
            >>> result = HiddenInfoExtractor.similarity_from_embeddings(a, b)
            >>> print(result["avg_similarity"])
        """
        # 统一为 2D: (num_layers, hidden_dim)
        if embedding_a.ndim == 1:
            embedding_a = embedding_a[np.newaxis, :]
        if embedding_b.ndim == 1:
            embedding_b = embedding_b[np.newaxis, :]

        # 显式 L2 归一化后再计算余弦相似度，防止调用方传入未归一化向量
        norm_a = np.linalg.norm(embedding_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(embedding_b, axis=1, keepdims=True)
        dot_product = (embedding_a * embedding_b).sum(axis=1)
        cos_sim = dot_product / (norm_a.squeeze() * norm_b.squeeze())
        cos_sim = np.nan_to_num(cos_sim, nan=0.0)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        return {
            "layer_similarities": cos_sim.tolist(),
            "avg_similarity": float(cos_sim.mean()),
            "max_similarity": float(cos_sim.max()),
            "min_similarity": float(cos_sim.min()),
        }

    # ------------------------------------------------------------------
    # 相似度计算（实例方法：从句子直接计算）
    # ------------------------------------------------------------------

    def similarity(
        self,
        sentence_a: str,
        sentence_b: str,
        output_attentions: bool = True,
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        normalize: bool = True,
        show_progress: bool = True, 
    ) -> dict:
        """计算两个句子在各层的余弦相似度（端到端）。

        内部流程: 句子 → embed() → similarity_from_embeddings()

        Args:
            sentence_a: 句子 A
            sentence_b: 句子 B
            output_attention: 是否输出注意力(weighted_mean需要)
            pooling:    池化策略
            layers:     指定层（None = 全部层）
            normalize:  是否 L2 归一化

        Returns:
            dict: {"layer_similarities": [...], "avg_similarity": float, "max_similarity": float, "min_similarity": float}
        """
        embeddings = self.embed(
            [sentence_a, sentence_b],
            pooling=pooling,
            output_attentions=output_attentions,
            layers=layers,
            normalize=normalize,
            return_numpy=True,
            show_progress=show_progress, 
        )
        # (num_layers, 2, hidden_dim)
        a = embeddings[:, 0, :]
        b = embeddings[:, 1, :]

        return HiddenInfoExtractor.similarity_from_embeddings(a, b)

    def pairwise_similarity_matrix(
        self,
        sentences: list[str],
        output_attentions: bool = True, 
        batch_sizes: int = 8, 
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """计算句子列表两两之间的相似度矩阵。

        Args:
            sentences: 句子列表
            output_attentions：是否输出注意力(weighted_mean需要)
            batch_size: 批处理大小
            pooling:   池化策略
            layers:    层列表（None = 全部层平均）
            normalize: 是否 L2 归一化

        Returns:
            similarity_matrix: shape = (num_sentences, num_sentences)
        """
        embeddings = self.embed(
            sentences,
            pooling=pooling,
            layers=layers,
            normalize=normalize,
            return_numpy=True,
        )
        # (num_layers, num_sentences, hidden_dim) → 跨层平均
        sentence_embeddings = embeddings.mean(axis=0)
        # 显式 L2 归一化后再计算余弦相似度矩阵
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        normed = sentence_embeddings / np.maximum(norms, 1e-12)
        sim_matrix = normed @ normed.T
        return np.clip(sim_matrix, -1.0, 1.0)

    # ------------------------------------------------------------------
    # 模型信息
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """返回模型的层数等关键架构信息。

        不需要执行前向传播，直接从模型配置中读取。

        Returns:
            dict: {
                "model_path": str — 模型的路径
                "model_type": str — 模型类型
                "hidden_size": int — 隐藏维度
                "num_parameters": int — 模型参数总数
                "device": str — 设备信息
                "vocab_size": int — 词表大小
                "pad_token": str — 填充token
                "eos_token": str — 结束token
                "num_hidden_layers": int — transformer 层数（不含 embedding 层）
                "num_hidden_states": int — hidden_states 总层数（embedding + transformer 层）
                "num_attention_heads": int — 注意力头数
                "num_key_value_heads": int — KV 头数（GQA/MQA 模型）
            }
        """
        config = self.model.config
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(config, "n_layer", None)

        num_kv_heads = getattr(config, "num_key_value_heads", None)

        return {
            "model_path": self.model_path,
            "model_type": self.model.config.model_type,
            "hidden_size": self.model.config.hidden_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "num_hidden_layers": num_layers,
            "num_hidden_states": num_layers + 1 if num_layers is not None else None,
            "hidden_size": getattr(config, "hidden_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": num_kv_heads,
        }


# ---------------------------------------------------------------------------
# 注册内置池化策略
# ---------------------------------------------------------------------------

HiddenInfoExtractor._POOLING_REGISTRY = {
    "mean": HiddenInfoExtractor._pool_mean,
    "last_token": HiddenInfoExtractor._pool_last_token,
    "eos_token": HiddenInfoExtractor._pool_eos_token,
    "max": HiddenInfoExtractor._pool_max,
    "weighted_mean": HiddenInfoExtractor._pool_weighted_mean,
}