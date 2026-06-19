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

PoolingStrategy = Literal["mean", "last_token", "eos_token", "max", "weighted_mean", "cls_token"]
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            output_hidden_states=True,
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
        outputs = self.model(**inputs, output_attentions=output_attentions)
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
    # 池化辅助方法
    # ==================================================================

    @staticmethod
    def _pool_single_layer(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy: PoolingStrategy = "mean",
        attentions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """对单层 token-level 隐藏状态执行池化，得到句向量。

        Args:
            hidden_states:  shape = (batch_size, seq_len, hidden_dim)
            attention_mask: shape = (batch_size, seq_len) —— 1 表示真实 token，0 表示 padding
            strategy:       池化策略
            attentions:     注意力权重 (batch, num_heads, seq_len, seq_len)，仅 weighted_mean 需要

        Returns:
            句向量 (batch_size, hidden_dim)
        """
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

        if strategy == "mean":
            masked = hidden_states * mask_expanded
            summed = masked.sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            return summed / counts

        elif strategy == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lengths, :]

        elif strategy == "eos_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, seq_lengths, :]

        elif strategy == "cls_token":
            return hidden_states[:, 0, :]

        elif strategy == "max":
            masked = hidden_states * mask_expanded + (1 - mask_expanded) * -1e9
            return masked.max(dim=1).values

        elif strategy == "weighted_mean":
            if attentions is None:
                raise ValueError(
                    "weighted_mean 策略需要传入注意力权重。"
                    "请确保模型以 output_attentions=True 加载，并在 encode() 中设置 output_attentions=True。"
                )
            last_attn = attentions[-1]                          # (batch, num_heads, seq_len, seq_len)
            avg_attn = last_attn.mean(dim=1)                    # (batch, seq_len, seq_len)
            weights = avg_attn.sum(dim=-1)                      # (batch, seq_len)
            weights = weights * attention_mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(
                f"不支持的池化策略: {strategy}。"
                f"可选: mean, last_token, eos_token, max, weighted_mean, cls_token"
            )

    # ==================================================================
    # 句嵌入提取
    # ==================================================================

    @staticmethod
    def extract_sentence_embedding(
        hidden_states: tuple,
        attention_mask: torch.Tensor,
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        attentions: Optional[tuple] = None,
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """从模型输出的 hidden_states 中提取句嵌入向量。

        对每一层执行池化，将 (batch, seq_len, hidden_dim) 聚合为 (batch, hidden_dim)，
        最终堆叠为 (num_layers, batch, hidden_dim)。

        Args:
            hidden_states:  tuple of (batch, seq_len, hidden_dim)，每层一个 tensor
            attention_mask: (batch, seq_len) —— 1 表示真实 token，0 表示 padding
            pooling:        池化策略，可选: mean, last_token, eos_token, max, weighted_mean, cls_token
            layers:         指定要返回的层索引（None = 全部层）。0 为 embedding 层，-1 为最后一层
            attentions:     attention weights tuple，仅 weighted_mean 策略需要
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
            attn_for_layer = attentions if pooling == "weighted_mean" else None
            sent_vec = HiddenInfoExtractor._pool_single_layer(
                hs, attention_mask, pooling, attentions=attn_for_layer
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

        embeddings_tensor = embeddings_tensor.cpu()

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
                attn_for_layer = attns if pooling == "weighted_mean" else None
                sent_vec = self._pool_single_layer(
                    hs, attn_mask, pooling, attentions=attn_for_layer
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

        embeddings_tensor = embeddings_tensor.cpu()

        if return_numpy:
            return embeddings_tensor.numpy()
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

        cos_sim = (embedding_a * embedding_b).sum(axis=1)
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
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        normalize: bool = True,
    ) -> dict:
        """计算两个句子在各层的余弦相似度（端到端）。

        内部流程: 句子 → embed() → similarity_from_embeddings()

        Args:
            sentence_a: 句子 A
            sentence_b: 句子 B
            pooling:    池化策略
            layers:     指定层（None = 全部层）
            normalize:  是否 L2 归一化

        Returns:
            dict: {"layer_similarities": [...], "avg_similarity": float, "max_similarity": float, "min_similarity": float}
        """
        embeddings = self.embed(
            [sentence_a, sentence_b],
            pooling=pooling,
            layers=layers,
            normalize=normalize,
            return_numpy=True,
        )
        # (num_layers, 2, hidden_dim)
        a = embeddings[:, 0, :]
        b = embeddings[:, 1, :]

        return HiddenInfoExtractor.similarity_from_embeddings(a, b)

    def pairwise_similarity_matrix(
        self,
        sentences: list[str],
        pooling: PoolingStrategy = "mean",
        layers: Optional[list[int]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """计算句子列表两两之间的相似度矩阵。

        Args:
            sentences: 句子列表
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
        sim_matrix = sentence_embeddings @ sentence_embeddings.T
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