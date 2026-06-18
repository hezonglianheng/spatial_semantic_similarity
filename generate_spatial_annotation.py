import call_api

import re
import json
import os
import sys
import time
import argparse
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import Optional, Any

# ==========

# prompt
SYSTEM_PROMPT = """
你是一位语言中空间问题的专家。请根据用户的问题，提供准确的空间信息。
"""

USER_PROMPT = """
你是一位语言中空间问题的专家。下面有两个只在一个方位词或者趋向动词上存在差异的句子。请模仿示例的格式，为我标注与差异部分相关的目标物和参照物。注意，只需要标注存在方位词或趋向动词差异部分的目标物和参照物，不要标注其他信息。返回的信息需要是标准JSON格式。\n

示例：
sentence1: 他穿街过巷，来到段公馆的后花园外，只听从高墙后飘出一阵笙、管、笛、萧的乐声和缠绵柔婉的《长生殿》歌声。
sentence2: 他穿街过巷，来到段公馆的后花园外，只听从高墙外飘出一阵笙、管、笛、萧的乐声和缠绵柔婉的《长生殿》歌声。
annotation: {
    "sentence1": {
        "target": "歌声", 
        "reference": "高墙"
    }, 
    "sentence2": {
        "target": "歌声", 
        "reference": "高墙"
    }
}

sentence1: 在罗马动物园的猴山里，老猴王死去了，早已跃跃欲试的一只大猴子继任为新猴王。当老猴王仍在世时，它还有所顾忌，现在却显示出无比的残暴，不但吓得群猴日夜惶惶，而且还不许它们吃饱。它自己虽已饱得无可再饱，但还不让别只吃，它把地上所有的食物集拢成一大堆，然后一屁股坐下去，怡然自得。
sentence2: 在罗马动物园的猴山里，老猴王死去了，早已跃跃欲试的一只大猴子继任为新猴王。当老猴王仍在世时，它还有所顾忌，现在却显示出无比的残暴，不但吓得群猴日夜惶惶，而且还不许它们吃饱。它自己虽已饱得无可再饱，但还不让别只吃，它把地上所有的食物集拢成一大堆，然后一屁股坐上去，怡然自得。
annotation: {
    "sentence1": {
        "target": "新猴王",
        "reference": "食物"
    },
    "sentence2": {
        "target": "新猴王",
        "reference": "食物"
    }
}

问题：
sentence1: {sentence1}
sentence2: {sentence2}
annotation: 
"""

# ==========

# Thread-safe printing
_print_lock = threading.Lock()

def _safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with _print_lock:
        print(*args, **kwargs)

def single_call(sentence1: str, sentence2: str, model: str, interval: float = .2) -> Optional[str]:
    """
    调用API生成空间标注

    Args:
        sentence1 (str): 第一个句子
        sentence2 (str): 第二个句子
        model (str): 使用的模型
        interval (float): 随机休眠时间间隔，单位为秒，默认为0.2

    Returns:
        Optional[str]: API响应结果，如果失败则返回None
    """
    # 填充USER_PROMPT（使用 replace 避免句子中的花括号导致 KeyError）
    user_prompt = USER_PROMPT.replace("{sentence1}", sentence1).replace("{sentence2}", sentence2)
    # 随机休眠
    time.sleep(random.uniform(0, interval))
    # 调用API
    response = call_api.query(
        prompt=user_prompt,
        model=model,
        system_prompt=SYSTEM_PROMPT,
    )

    return response

def response_parse(response: Optional[str]):
    # 解析API响应
    if response is None:
        return {"error": "Empty response from API"}
    try:
        respose_dict: dict[str, dict[str, str]] = json.loads(response)
        # 检查字段的存在性
        if "sentence1" not in respose_dict or "sentence2" not in respose_dict:
            return {"error": "Missing required fields"}
        sentence1 = respose_dict["sentence1"]
        sentence2 = respose_dict["sentence2"]
        if "target" not in sentence1 or "reference" not in sentence1:
            return {"error": "Missing required fields in sentence1"}
        elif not isinstance(sentence1["target"], str) or not isinstance(sentence1["reference"], str):
            return {"error": "Invalid data types in sentence1"}
        if "target" not in sentence2 or "reference" not in sentence2:
            return {"error": "Missing required fields in sentence2"}
        elif not isinstance(sentence2["target"], str) or not isinstance(sentence2["reference"], str):
            return {"error": "Invalid data types in sentence2"}
        return {
            "sentence1": {
                "target": sentence1["target"],
                "reference": sentence1["reference"]
            },
            "sentence2": {
                "target": sentence2["target"],
                "reference": sentence2["reference"]
            }
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}

def sentence_pair_query(idx: int, sentence1: str, sentence2: str, model: str, interval: float = .2):
    retry_time = 0
    while retry_time < 5:
        try:
            _safe_print(f"[信息] 开始处理第{idx}个句子对(第{retry_time + 1}次尝试).")
            response = single_call(sentence1, sentence2, model, interval)
            parsed_response = response_parse(response)
            if "error" not in parsed_response:
                return parsed_response
        except Exception as e:
            _safe_print(f"[错误] 发生错误: {e}.")
        retry_time += 1
    _safe_print(f"[错误] 第{idx}个句子对处理失败.")
    return None

def read_sentences_from_file(file_path: str) -> list[dict[str, Any]]:
    """读取句子对文件.
    
    Args:
        file_path (str): 句子对文件的路径

    Returns:
        list[dict[str, Any]]: 句子对列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    assert isinstance(data, list), "Expected a list of dictionaries"
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"
    assert all("sentence1" in item and "sentence2" in item for item in data), "All items must contain 'sentence1' and 'sentence2' keys"
    return data

def query_list(sentence_list: list[dict[str, Any]], model: str, interval: float = .2):
    """处理句子序列

    Args:
        sentence_list (list[dict[str, Any]]): _description_
        model (str): _description_
        interval (float, optional): _description_. Defaults to .2.

    Returns:
        _type_: _description_
    """
    results = []
    cpu_count = os.cpu_count() or 1
    print("[信息] 开始处理句子序列，使用 {} 个CPU核心.".format(max(cpu_count // 2, 1)))
    with ThreadPoolExecutor(max_workers=max(cpu_count // 2, 1)) as executor:
        futures = [
            executor.submit(sentence_pair_query, idx, item["sentence1"], item["sentence2"], model, interval)
            for idx, item in enumerate(sentence_list)
        ]
        results = [future.result() for future in futures]
    # 将results拼接到原始数据中
    for idx, item in enumerate(sentence_list):
        item["annotation"] = results[idx]
    return sentence_list

def main():
    parser = argparse.ArgumentParser(description="用大模型对异形同义句子对做方位信息标注")
    parser.add_argument("input_file" , help="输入的句子对文件，需要是json文件")
    parser.add_argument("--output_dir", "-o", default="./spatial_info_annotation", help="输出的标注结果目录")
    parser.add_argument("--model", "-m", help="使用的模型")
    parser.add_argument("--interval", "-t", type=float, default=0.2, help="API调用间隔时间")
    args = parser.parse_args()

    # 1. 读取文件
    sentence_list = read_sentences_from_file(args.input_file)

    # 2. 处理句子序列
    annotated_sentence_list = query_list(sentence_list, args.model, args.interval)

    # 3. 保存结果
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(args.output_dir, f"spatial_dataset_{args.model}_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotated_sentence_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()