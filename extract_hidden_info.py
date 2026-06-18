# encoding: utf8

import os
import sys
import json
import argparse
from typing import Optional, Union, Literal

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code,trust_remote_code=trust_remote_code).to(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code,output_hidden_state=True, output_hidden_attention=True,**kwargs).to(device)
        self.model.eval()