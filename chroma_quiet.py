"""chromadb 客户端设置 + 遥测治理（独立小模块，**所有入口都应显式 import 它**）

为什么单独一个模块：
    之前这两件事放在 vector_stores.py 里，靠 import 链间接生效
    （server.py / scripts/ingest.py 自己 new chromadb.Client 时，是否被屏蔽取决于
     "谁先 import 了 vector_stores"）。这属于隐藏的时序依赖，容易在某次重构后静默失效。
    现在把它独立出来，入口文件在**最顶部显式 import**，语义明确、不依赖巧合。

**遥测屏蔽的意图（不要"修好"它，我们完全不需要遥测）**：
    chromadb 0.4.24 的 posthog 适配层用 3 个位置参数调 posthog.capture，
    新版 posthog 的 capture 只接受 1 个 → 每次上报都在签名校验阶段抛
        TypeError: capture() takes 1 positional argument but 3 were given
    然后被 logger.error 打出来。而 Settings(anonymized_telemetry=False) 只是把
    posthog.disabled 置 True，**挡不住这行 logger.error**（异常发生在真正发送之前）。
    所以这里直接把该 logger 提到 CRITICAL：既不发送，也不再刷日志。

★ 持久化必须显式打开：
    langchain_chroma 只有在**不传** client_settings 时才自己构造
        Settings(is_persistent=True, persist_directory=...)
    一旦传入 client_settings 就照用不误、**不会补 is_persistent**，
    而 chromadb 的 Settings 默认 is_persistent=False → 退化成"进程内内存库"：
    入库进程内 count() 正常，换进程就查不到集合（数据根本没落盘）。
"""

import logging

from chromadb.config import Settings

import config_data as config


def silence_telemetry() -> None:
    """把 chromadb 的遥测 logger 提到 CRITICAL（导入本模块即生效）"""
    # 故意把该 logger 提到 CRITICAL：chromadb 0.4.24 的 posthog 适配层
    # 用 3 个位置参数调 posthog.capture，新版 posthog 只接受 1 个，
    # 异常发生在签名校验阶段，anonymized_telemetry=False 挡不住这行 logger.error。
    # ★ 不要"修好"它——我们完全不需要遥测。
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


# 导入即生效：必须在任何 chromadb 客户端被创建之前执行
silence_telemetry()


def build_settings() -> Settings:
    """统一的 chromadb 客户端设置（含持久化 + 关遥测）"""
    return Settings(
        anonymized_telemetry=False,     # 阻止真正上报
        is_persistent=True,             # ★ 见模块文档：漏了它就是进程内内存库
        persist_directory=config.persist_directory,
    )


# 共享实例：Chroma 封装、脚本、服务端都用这一份，避免各处 Settings 走样
CHROMA_SETTINGS = build_settings()
