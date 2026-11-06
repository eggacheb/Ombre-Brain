"""入库后预脱水 —— 把第一次脱水提前到落桶时刻，焐热内容哈希缓存。

/breath-hook（SessionStart HTTP 钩子）每次被拉取都会对选中的浮现桶**现场**
脱水：串行 LLM 调用，上限 hooks 的 max_dehydrate_calls（默认 8）× 单次
dehydrate_timeout_seconds（默认 12s）。dehydrator 自带按内容哈希的 SQLite
缓存（dehydrate() 先查后调），但活跃 vault 上新桶源源不断、换模型还会让
缓存整体作废（key 含 model），缓存长期半冷。当消费方以「每条消息一次」
的高频拉取（如 chatgpt-plugin 的上下文注入）时，插件侧几秒的预算经常不够，
浮现整轮被跳过。

本模块在桶落盘的后台任务里对该桶调一次 dehydrate：结果写入现有缓存，
拉取路径一行不改。只对会进入浮现池的桶预脱——hooks.py 的两个主池
（pinned + unresolved）共同排除的类型不做；digested / dont_surface 等
运行时状态**不**在这里过滤：它们只是「当前不浮现」，桶恢复浮现时缓存
照样命中，提前脱只是少量提前花费，不值得为它复刻一份 surface_policy
增加漂移面。
"""

from ombrebrain.storage.letter_lock import is_letter_bucket

from . import _runtime as rt

# web/hooks.py breath_hook 的两个主池共同的排除项：unresolved 池排除这些类型，
# pinned 池只看 pinned 标记（pinned 桶可能落在 permanent/，靠标记放行）；两池
# 都排除 protected 与 letter（后者是生命周期改写，按 is_letter_bucket 识别——
# 直接 import 公共实现，避免这里再抄一份判定逻辑漂移）。类型在其中且未 pinned
# 的桶永远不会被浮现场景选中，预脱它们纯属白花 API 钱。
_NON_SURFACE_TYPES = {"permanent", "feel", "plan", "letter", "self", "i"}


def _truthy(value) -> bool:
    """与 web/hooks.py 的 _truthy 同语义：frontmatter 读回的布尔可能是字符串。"""
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


async def predehydrate_for_surface(bucket_id: str) -> bool:
    """后台预脱一个桶，返回是否真的调用了 dehydrator。

    调用方应当 `asyncio.create_task(...)`，不要 await——写入返回不等这个。
    失败只记日志：预脱水是性能优化不是正确性依赖，拉取路径对未命中缓存
    本来就有现场脱水兜底（dehydrate 自身还会降级到原文截断）。
    """
    try:
        bucket = await rt.bucket_mgr.get(bucket_id)
        if not bucket:
            return False
        metadata = bucket.get("metadata") or {}
        if _truthy(metadata.get("protected")) or is_letter_bucket(bucket):
            return False
        bucket_type = str(metadata.get("type") or "")
        if not _truthy(metadata.get("pinned")) and bucket_type in _NON_SURFACE_TYPES:
            return False
        content = str(bucket.get("content") or "")
        if not content.strip():
            return False
        await rt.dehydrator.dehydrate(content)
        return True
    except Exception as exc:  # noqa: BLE001 - fire-and-forget，绝不影响写入
        rt.logger.warning(
            f"predehydrate failed / 预脱水失败: {type(exc).__name__}: {exc}"
        )
        return False
