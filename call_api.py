"""
API 调用脚本 —— 使用 OpenAI 兼容接口发送请求并获取结果。

功能：
  1. 从 api_config.json 读取 API 配置，每个模型拥有独立的 base_url、api_key 等参数
  2. 通过命令行参数或函数调用指定模型和传入 prompt 文本
  3. 内置指数退避重试、超时控制、异常处理等容错机制
  4. 支持流式和非流式两种响应模式
  5. 可作为模块被其他程序 import 使用（核心函数：query()）

用法：
  # 命令行：直接传入 prompt 文本
  python call_api.py "你好，请介绍一下你自己"

  # 命令行：指定模型
  python call_api.py --model deepseek-chat "解释什么是空间语义相似性"

  # 命令行：从文件读取 prompt
  python call_api.py --file input.txt

  # 作为模块被其他程序调用：
  from call_api import query
  result = query(prompt="你好，请介绍一下你自己", model="gpt-4o")
"""

import json
import time
import os
import sys
import argparse
import random
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError

# ---------------------------------------------------------------------------
# 1. 加载配置文件
# ---------------------------------------------------------------------------

# 模型参数的硬编码默认值（仅在配置文件中某字段缺失时回退）
_MODEL_DEFAULTS: dict = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "timeout": 60,
}


def load_config(config_path: str = "api_config.json") -> dict:
    """
    从 JSON 配置文件中读取 API 参数。

    配置文件格式 —— api_config.json 的结构是一个模型字典，
    每个 key 是模型名称，value 是该模型的完整配置：

    {
        "default_model": "gpt-4o",
        "models": {
            "gpt-4o": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-your-openai-key-here",
                "timeout": 60,
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "deepseek-chat": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-your-deepseek-key-here",
                "timeout": 90,
                "temperature": 0.7,
                "max_tokens": 4096
            }
        }
    }

    参数:
        config_path: 配置文件路径

    返回:
        原始配置字典（default_model + models）
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"配置文件 '{config_path}' 不存在，请先创建该文件并填入 API 参数。\n"
            f"格式参考本函数文档字符串中的示例。"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # models 字段必须存在且非空
    if "models" not in config or not isinstance(config["models"], dict) or len(config["models"]) == 0:
        raise ValueError(
            "配置文件中缺少 'models' 字段或为空。\n"
            "models 应当是一个字典，key 为模型名称，value 为该模型的完整参数。"
        )

    # default_model：未指定则自动取 models 中的第一个
    if "default_model" not in config or not config["default_model"]:
        config["default_model"] = list(config["models"].keys())[0]
    elif config["default_model"] not in config["models"]:
        available = ", ".join(config["models"].keys())
        raise ValueError(
            f"default_model '{config['default_model']}' 不在 models 中。\n"
            f"可用的模型: {available}"
        )

    return config


# ---------------------------------------------------------------------------
# 2. 解析模型配置
# ---------------------------------------------------------------------------

def get_model_config(config: dict, model_name: str | None = None) -> dict:
    """
    从配置中提取指定模型的完整参数。

    每个模型必须显式配置 base_url 和 api_key；
    timeout / temperature / max_tokens 若缺失则使用硬编码默认值。

    参数:
        config:     load_config() 返回的原始配置字典
        model_name: 模型名称（为 None 则使用 default_model）

    返回:
        包含以下字段的字典：
        - name:         模型名称
        - base_url:     API 端点地址
        - api_key:      API 密钥
        - timeout:      请求超时时间（秒）
        - temperature:  采样温度
        - max_tokens:   最大生成 token 数
    """
    if model_name is None:
        model_name = config["default_model"]

    models = config["models"]
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(
            f"模型 '{model_name}' 不在配置文件的 models 中。\n"
            f"可用的模型: {available}\n"
            f"默认模型: {config['default_model']}"
        )

    raw = models[model_name]
    resolved = {"name": model_name}

    # ---- 必填字段 ----
    resolved["base_url"] = raw.get("base_url", "")
    resolved["api_key"] = raw.get("api_key", "")

    if not resolved["base_url"]:
        raise ValueError(f"模型 '{model_name}' 缺少必填字段 'base_url'。")
    if not resolved["api_key"]:
        raise ValueError(f"模型 '{model_name}' 缺少必填字段 'api_key'。")

    # ---- 可选字段：缺失时使用硬编码默认值 ----
    resolved["timeout"]     = raw.get("timeout",     _MODEL_DEFAULTS["timeout"])
    resolved["temperature"] = raw.get("temperature", _MODEL_DEFAULTS["temperature"])
    resolved["max_tokens"]  = raw.get("max_tokens",  _MODEL_DEFAULTS["max_tokens"])

    return resolved


# ---------------------------------------------------------------------------
# 3. 创建 OpenAI 客户端
# ---------------------------------------------------------------------------

def create_client(model_cfg: dict) -> OpenAI:
    """
    根据已解析的模型配置创建 OpenAI 客户端实例。

    参数:
        model_cfg: get_model_config() 返回的模型参数字典

    返回:
        OpenAI 客户端实例
    """
    return OpenAI(
        base_url=model_cfg["base_url"],  # 每个模型连接各自配置的 API 端点
        api_key=model_cfg["api_key"],    # 每个模型使用各自配置的密钥
        timeout=model_cfg["timeout"],    # 每个模型使用各自配置的超时时间
    )


# ---------------------------------------------------------------------------
# 4. 带重试机制的 API 调用（内部核心函数）
# ---------------------------------------------------------------------------

def _chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    stream: bool = False,
) -> str | None:
    """
    [内部函数] 发送聊天补全请求，内置指数退避重试机制。

    重试策略:
      - 第 1 次失败后等待 initial_backoff 秒
      - 第 2 次失败后等待 initial_backoff * backoff_factor 秒
      - 第 3 次失败后等待 initial_backoff * backoff_factor^2 秒
      - 以此类推，直到达到 max_retries 次

    可重试的错误类型:
      - 速率限制 (429)
      - 服务器内部错误 (5xx)
      - 网络连接错误

    不可重试的错误类型（立即抛出）:
      - 认证错误 (401)
      - 参数错误 (400)

    返回:
        API 返回的文本内容；若所有重试均失败则返回 None
    """
    last_error = None

    for attempt in range(1, max_retries + 2):  # 1 次正常请求 + max_retries 次重试
        try:
            print(f"[第 {attempt} 次尝试] 正在请求模型 '{model}'...")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            # ---------- 处理流式响应 ----------
            if stream:
                full_content = ""
                print("[流式输出] ", end="", flush=True)
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_piece = chunk.choices[0].delta.content
                        print(content_piece, end="", flush=True)
                        full_content += content_piece
                print()
                return full_content

            # ---------- 处理非流式响应 ----------
            content = response.choices[0].message.content
            print("[请求成功] 已收到完整响应。")
            return content

        # ==================== 异常分类处理 ====================

        except RateLimitError as e:
            # 429 Too Many Requests —— 速率限制
            last_error = e
            wait_time = _calculate_wait_time(attempt, initial_backoff, backoff_factor, e)
            print(f"[速率限制] 请求过于频繁，将在 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)

        except APITimeoutError as e:
            # 请求超时 —— 可能服务器负载高
            last_error = e
            wait_time = _calculate_wait_time(attempt, initial_backoff, backoff_factor, e)
            print(f"[超时] 请求超时，将在 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)

        except APIConnectionError as e:
            # 网络连接错误 —— 可能是临时网络波动
            last_error = e
            wait_time = _calculate_wait_time(attempt, initial_backoff, backoff_factor, e)
            print(f"[连接错误] 无法连接到 API 服务器，将在 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)

        except APIError as e:
            # 其他 API 错误 —— 根据 HTTP 状态码决定是否重试
            last_error = e
            status_code = getattr(e, "status_code", None)

            if status_code is not None and 500 <= status_code < 600:
                # 5xx 服务器错误 —— 可重试
                wait_time = _calculate_wait_time(attempt, initial_backoff, backoff_factor, e)
                print(f"[服务器错误 {status_code}] 服务器内部错误，将在 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            elif status_code == 429:
                # 部分代理服务将 429 包装为普通 APIError
                wait_time = _calculate_wait_time(attempt, initial_backoff, backoff_factor, e)
                print(f"[速率限制] 请求过于频繁，将在 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 不可重试的错误（如 400 Bad Request, 401 Unauthorized）
                print(f"[不可重试的错误] 状态码: {status_code}, 消息: {e}")
                raise

        except AuthenticationError as e:
            # 401 —— API Key 无效，不重试
            print(f"[认证失败] API Key 无效或已过期，请检查 api_config.json 中该模型的 api_key。")
            raise

        except KeyboardInterrupt:
            print("\n[中断] 用户取消了请求。")
            raise

        except Exception as e:
            # 未知异常 —— 保守起见，记录并重试
            last_error = e
            wait_time = _calculate_wait_time(attempt, initial_backoff, backoff_factor, None)
            print(f"[未知异常] {type(e).__name__}: {e}，将在 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)

    # 所有重试均已用尽
    print(f"\n[失败] 已重试 {max_retries} 次，仍然无法获得有效响应。")
    if last_error:
        print(f"最后一次错误: {type(last_error).__name__}: {last_error}")
    return None


def _calculate_wait_time(
    attempt: int,
    initial_backoff: float,
    backoff_factor: float,
    error: Exception | None,
) -> float:
    """
    计算退避等待时间（指数退避 + 随机抖动）。

    如果服务端返回了 Retry-After 头，则优先使用其建议值。

    参数:
        attempt:         当前第几次尝试（从 1 开始）
        initial_backoff: 初始退避等待时间（秒）
        backoff_factor:  退避倍数
        error:           捕获的异常对象（可能为 None）

    返回:
        建议等待的秒数
    """
    # 服务端 Retry-After 头优先
    if error is not None:
        retry_after = _extract_retry_after(error)
        if retry_after is not None:
            return retry_after

    # 指数退避: initial_backoff * backoff_factor^(attempt-1)
    wait_time = initial_backoff * (backoff_factor ** (attempt - 1))

    # 随机抖动 ±25%，避免"惊群效应"
    jitter = wait_time * 0.25 * (2 * random.random() - 1)
    return max(0, wait_time + jitter)


def _extract_retry_after(error: Exception) -> float | None:
    """尝试从异常对象中提取 Retry-After 头的值（秒）。"""
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
    return None


# ---------------------------------------------------------------------------
# 5. 面向外部调用者的核心接口 —— query()
# ---------------------------------------------------------------------------

def query(
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    config_path: str = "api_config.json",
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    stream: bool = False,
    # 以下参数若传入则会覆盖配置文件中的模型参数
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """
    发送单个 prompt 到指定模型并返回响应文本。

    这是面向外部调用者的核心接口。其他 Python 程序可以直接 import 并调用此函数，
    无需关心内部的配置加载、客户端创建、重试等细节。

    使用示例:
        from call_api import query

        # 使用默认模型
        result = query(prompt="你好，请介绍一下你自己")

        # 指定模型（自动使用该模型在配置中的 base_url 和 api_key）
        result = query(prompt="解释空间语义相似性", model="deepseek-chat")

        # 带自定义系统提示词
        result = query(
            prompt="什么是空间语义相似性？",
            model="gpt-4o",
            system_prompt="你是一个空间信息科学领域的专家。请用中文回答。"
        )

    参数:
        prompt:           用户输入的提示文本（必填，由外部程序传入）
        model:            模型名称，为 None 则使用配置文件中的 default_model
        system_prompt:    系统提示词
        config_path:      配置文件路径
        max_retries:      最大重试次数
        initial_backoff:  初始退避等待时间（秒）
        backoff_factor:   退避倍数
        stream:           是否使用流式输出
        temperature:      采样温度，传入则覆盖配置文件中的值
        max_tokens:       最大生成 token 数，传入则覆盖配置文件中的值

    返回:
        API 返回的文本内容；若所有重试均失败则返回 None
    """
    raw_config = load_config(config_path)
    model_cfg = get_model_config(raw_config, model)

    # 调用者显式传入 → 覆盖配置文件中的值
    if temperature is not None:
        model_cfg["temperature"] = temperature
    if max_tokens is not None:
        model_cfg["max_tokens"] = max_tokens

    client = create_client(model_cfg)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    return _chat_completion(
        client=client,
        model=model_cfg["name"],
        messages=messages,
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        backoff_factor=backoff_factor,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# 6. 批量请求（带间隔控制）
# ---------------------------------------------------------------------------

def batch_query(
    prompts: list[str],
    *,
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    config_path: str = "api_config.json",
    interval: float = 1.0,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> list[str | None]:
    """
    批量发送多个 prompt，每个 prompt 之间自动间隔以控制请求频率。

    使用示例:
        from call_api import batch_query

        prompts = [
            "什么是空间语义相似性？",
            "空间相似性和语义相似性有什么区别？",
        ]
        results = batch_query(prompts, model="gpt-4o")

    参数:
        prompts:        待发送的 prompt 列表
        model:          模型名称，为 None 则使用默认模型
        system_prompt:  系统提示词
        config_path:    配置文件路径
        interval:       每次请求之间的固定等待时间（秒），防止触发速率限制
        temperature:    采样温度，传入则覆盖配置文件中的值
        max_tokens:     最大生成 token 数，传入则覆盖配置文件中的值
        **kwargs:       传递给 _chat_completion() 的其他参数

    返回:
        每个 prompt 对应的响应列表（失败项为 None）
    """
    raw_config = load_config(config_path)
    model_cfg = get_model_config(raw_config, model)
    client = create_client(model_cfg)

    if temperature is not None:
        model_cfg["temperature"] = temperature
    if max_tokens is not None:
        model_cfg["max_tokens"] = max_tokens

    results: list[str | None] = []
    total = len(prompts)

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n{'='*60}")
        print(f"[批量请求 {i}/{total}]")
        print(f"{'='*60}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = _chat_completion(
            client=client,
            model=model_cfg["name"],
            messages=messages,
            temperature=model_cfg["temperature"],
            max_tokens=model_cfg["max_tokens"],
            **kwargs,
        )
        results.append(result)

        if i < total:
            print(f"[频率控制] 等待 {interval:.1f} 秒后发送下一个请求...")
            time.sleep(interval)

    success_count = sum(1 for r in results if r is not None)
    print(f"\n{'='*60}")
    print(f"[批量请求完成] 共 {total} 条，成功 {success_count} 条，失败 {total - success_count} 条。")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# 7. 列出配置中所有可用模型
# ---------------------------------------------------------------------------

def list_models(config_path: str = "api_config.json") -> list[str]:
    """
    列出配置文件中所有可用的模型名称及其连接信息。

    参数:
        config_path: 配置文件路径

    返回:
        模型名称列表
    """
    raw_config = load_config(config_path)
    default = raw_config["default_model"]

    print(f"可用模型 ({len(raw_config['models'])} 个):")
    for name, cfg in raw_config["models"].items():
        tag = " [默认]" if name == default else ""
        base_url = cfg.get("base_url", "(未配置)")
        print(f"  - {name}{tag}")
        print(f"    base_url: {base_url}")

    return list(raw_config["models"].keys())


# ---------------------------------------------------------------------------
# 8. 命令行入口
# ---------------------------------------------------------------------------

def main():
    """
    命令行入口：解析参数，调用 query() 并输出结果。

    用法示例:
        python call_api.py "你好，请介绍一下你自己"
        python call_api.py --model deepseek-chat "解释空间语义相似性"
        python call_api.py --file input.txt
        python call_api.py --list-models
    """
    parser = argparse.ArgumentParser(
        description="使用 OpenAI 兼容 API 发送聊天补全请求",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python call_api.py "你好，请介绍一下你自己"
  python call_api.py --model deepseek-chat "解释空间语义相似性"
  python call_api.py --model gpt-4o --system "你是空间科学专家" "什么是空间语义相似性？"
  python call_api.py --file input.txt
  python call_api.py --list-models
        """,
    )

    # prompt 来源：命令行直接传入 或 从文件读取
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "prompt", nargs="?", default=None,
        help="要发送的 prompt 文本（直接在命令行中传入）",
    )
    prompt_group.add_argument(
        "--file", "-f", type=str, default=None,
        help="从文件中读取 prompt 文本",
    )

    parser.add_argument("--model", "-m", type=str, default=None,
                        help="指定要使用的模型名称（不指定则使用 default_model）")
    parser.add_argument("--system", "-s", type=str, default="You are a helpful assistant.",
                        help="系统提示词")
    parser.add_argument("--config", "-c", type=str, default="api_config.json",
                        help="配置文件路径（默认: api_config.json）")
    parser.add_argument("--temperature", "-t", type=float, default=None,
                        help="采样温度，覆盖配置文件中的值")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="最大生成 token 数，覆盖配置文件中的值")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="最大重试次数（默认: 5）")
    parser.add_argument("--stream", action="store_true", default=False,
                        help="启用流式输出")
    parser.add_argument("--list-models", action="store_true", default=False,
                        help="列出配置文件中所有可用模型并退出")

    args = parser.parse_args()

    if args.list_models:
        list_models(args.config)
        return

    # 获取 prompt 文本
    if args.file is not None:
        if not os.path.exists(args.file):
            print(f"[错误] 文件 '{args.file}' 不存在。")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            print(f"[错误] 文件 '{args.file}' 内容为空。")
            sys.exit(1)
        print(f"[从文件读取] {args.file} ({len(prompt)} 字符)")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        parser.print_help()
        print("\n[提示] 请提供 prompt 文本，例如:")
        print('  python call_api.py "你好，请介绍一下你自己"')
        sys.exit(1)

    print("=" * 60)
    print("  API 调用脚本")
    print("=" * 60)
    print(f"[模型] {args.model if args.model else '(使用默认模型)'}")
    print(f"[Prompt 长度] {len(prompt)} 字符")

    result = query(
        prompt=prompt,
        model=args.model,
        system_prompt=args.system,
        config_path=args.config,
        max_retries=args.max_retries,
        stream=args.stream,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if result:
        print("\n" + "=" * 60)
        print("  最终响应内容:")
        print("=" * 60)
        print(result)
        print("=" * 60)
    else:
        print("\n未能获得有效响应，请检查网络连接和 API 配置。")
        sys.exit(1)


if __name__ == "__main__":
    main()
