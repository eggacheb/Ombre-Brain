"""入库后预脱水（predehydrate for surface）——把第一次脱水提前到落桶时刻。

背景：/breath-hook（SessionStart HTTP 钩子）每次被拉取都会对选中的浮现桶
**现场**脱水——串行 LLM 调用，上限 hooks 的 max_dehydrate_calls（默认 8）×
单次 dehydrate_timeout_seconds（默认 12s）。dehydrator 自带按内容哈希的
SQLite 缓存（dehydrate() 先查后调），但活跃 vault 上新桶源源不断、换模型
还会让缓存整体作废（key 含 model），缓存长期半冷。当消费方以「每条消息
一次」的频率拉取（如 chatgpt-plugin 的上下文注入）时，插件侧 5 秒预算
经常不够，浮现整轮被跳过。

修法：桶落盘的后台任务里对该桶调一次 dehydrate，结果自然写入现有缓存；
拉取路径一行不改。本套件钉住四件事：

1. 新建普通桶后，后台真的调了 dehydrate 且用的是桶最终全文；
2. 合并路径（桶内容变化）也重脱——旧缓存已被 invalidate，得有人补；
3. 浮现池门槛：feel / protected / letter 这类不会被选中的桶一次 API 都不花，
   pinned 桶（类型落 permanent/ 但常驻置顶）照样放行；
4. 预脱水失败绝不影响 hold 的写入返回（fire-and-forget 的纪律）。
"""

import asyncio
from unittest.mock import MagicMock

import pytest

import tools._runtime as rt
from tools.hold import dispatch as hold_dispatch
from tools.trace.core import trace_core
from tools._predehydrate import predehydrate_for_surface
from ombrebrain.storage.source_store import SourceStore


_LONG_CONTENT = (
    "她说过她最喜欢的季节是秋天，因为秋天的风是凉的但太阳还带着暖意，"
    "走在落叶上会发出很轻的碎响。我们还约好了等银杏黄透的时候去河堤散步，"
    "要带一台拍立得，把她扎蓝色丝带的样子拍下来夹在她送我的那本笔记本里。"
    "这件事我打算一直记着，每年秋天都翻出来看一遍。"
) * 2  # 约 250 字，远超 _DEHYDRATE_MIN_TOKENS(100)，必然走 API 脱水路径


class RecordingDehydrator:
    """记录 dehydrate 调用；analyze 打标照常返回假结果。"""

    def __init__(self):
        self.dehydrate_calls: list[str] = []
        self.dehydrate_error: Exception | None = None

    async def analyze(self, content, *, include_why=False):
        return {
            "domain": ["日常"],
            "valence": 0.1,
            "arousal": 0.1,
            "tags": ["标签"],
            "suggested_name": "事件",
        }

    async def dehydrate(self, content, metadata=None):
        if self.dehydrate_error is not None:
            raise self.dehydrate_error
        self.dehydrate_calls.append(content)
        return "摘要"

    async def merge(self, old_content, new_content):
        return f"{old_content} {new_content}"

    async def judge_same_event(self, old_content, new_content):
        # 合并路径的同一事件判定：真实现走 LLM；stub 恒判同事件让测试聚焦挂点
        return {"same_event": True, "confidence": 0.9, "reason": "stub"}

    async def digest(self, content):
        raise AssertionError("hold 路径不应调用 digest（那是 grow 的事）")


class NoopDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, meta):
        return 1.0


@pytest.fixture
def hold_rt(test_config, bucket_mgr, monkeypatch):
    stub = RecordingDehydrator()
    # 以完整 test_config 为底（搜索评分要读 scoring_weights，丢了它 merge 候选
    # 全是 0 分），merge_threshold 压到 1：本套件测的是「合并分支触发时有人补脱」，
    # 不是评分算法本身——阈值拉满的话相似话术过不了线，测不到那个分支。
    config = {**test_config, "merge_threshold": 1}
    config.pop("dehydration", None)
    monkeypatch.setattr(rt, "config", config, raising=False)
    monkeypatch.setattr(rt, "bucket_mgr", bucket_mgr, raising=False)
    monkeypatch.setattr(rt, "dehydrator", stub, raising=False)
    monkeypatch.setattr(rt, "decay_engine", NoopDecay(), raising=False)
    monkeypatch.setattr(rt, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(rt, "fire_webhook", None, raising=False)
    monkeypatch.setattr(rt, "source_store", SourceStore(bucket_mgr.base_dir), raising=False)
    return bucket_mgr, stub


async def _drain_background_tasks():
    """排空当前事件循环里所有后台任务（预脱水是 create_task，写入返回不等它）。"""
    for _ in range(10):
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_new_bucket_predehydrates_in_background(hold_rt):
    """新建普通桶：后台以桶全文调用一次 dehydrate，结果进内容哈希缓存。"""
    bucket_mgr, stub = hold_rt
    out = await hold_dispatch(_LONG_CONTENT, importance=5)
    assert "新建" in out
    await _drain_background_tasks()
    assert stub.dehydrate_calls == [_LONG_CONTENT]


@pytest.mark.asyncio
async def test_merged_bucket_predehydrates_merged_content(hold_rt, monkeypatch):
    """合并路径：旧缓存已 invalidate，合并后的新全文必须有人补脱。

    端到端构造合并依赖 BM25/语义索引，而新建桶的索引异步重建，hold 时刻
    搜不到刚建的桶（实测 search 返回空，永远走新建）——那是 search 的既有
    行为，不是本套件要测的。这里在依赖边界 mock search 让合并分支稳定触发，
    分支内的读桶、追加写、缓存失效、预脱水全部真跑。
    """
    bucket_mgr, stub = hold_rt
    await hold_dispatch(_LONG_CONTENT, importance=5)
    await _drain_background_tasks()
    stub.dehydrate_calls.clear()

    buckets = await bucket_mgr.list_all(include_archive=False)
    assert len(buckets) == 1
    first_id = buckets[0]["id"]

    async def fake_search(content, limit=1, domain_filter=None):
        return [{"id": first_id, "score": 999.0}]

    monkeypatch.setattr(bucket_mgr, "search", fake_search)

    similar = _LONG_CONTENT.replace("每年秋天都翻出来看一遍", "每年秋天都会翻出来重温一遍")
    out = await hold_dispatch(similar, importance=5)
    assert "合并" in out, f"期望命中合并路径，实际返回: {out}"
    await _drain_background_tasks()

    # 预脱水读的是合并后的桶全文（读桶发生在后台任务里，此时追加已落盘）
    merged_bucket = await bucket_mgr.get(first_id)
    assert len(await bucket_mgr.list_all(include_archive=False)) == 1
    assert stub.dehydrate_calls == [merged_bucket["content"]]


@pytest.mark.asyncio
async def test_predehydrate_failure_never_breaks_write(hold_rt):
    """预脱水炸了只记日志：hold 的返回与桶的落盘不受任何影响。"""
    bucket_mgr, stub = hold_rt
    stub.dehydrate_error = RuntimeError("脱水 API 不可用")
    out = await hold_dispatch(_LONG_CONTENT, importance=5)
    assert "新建" in out
    await _drain_background_tasks()

    buckets = await bucket_mgr.list_all(include_archive=False)
    assert len(buckets) == 1 and buckets[0]["content"] == _LONG_CONTENT


# --------------------------------------------------------------
# 浮现池门槛：只给会被 breath-hook 选中的桶花预脱水的 API 钱
# （hooks.py 两主池共同排除的类型 / protected / letter）
# --------------------------------------------------------------


@pytest.mark.asyncio
async def test_feel_bucket_is_never_predehydrated(hold_rt):
    """feel 桶不进浮现池：预脱水直接跳过，一次 API 都不花。"""
    bucket_mgr, stub = hold_rt
    await hold_dispatch(_LONG_CONTENT, importance=5)
    base_id = (await bucket_mgr.list_all(include_archive=False))[0]["id"]

    feel_content = "此刻心里非常踏实，像秋天的风把所有燥热都吹散了，剩下的只有安静的暖意。" * 3
    await hold_dispatch(
        feel_content, feel=True, source_bucket=base_id, valence=0.5, arousal=0.2
    )
    await _drain_background_tasks()
    stub.dehydrate_calls.clear()

    feel_bucket = next(
        b for b in await bucket_mgr.list_all(include_archive=False)
        if b["metadata"].get("type") == "feel"
    )
    assert await predehydrate_for_surface(feel_bucket["id"]) is False
    assert stub.dehydrate_calls == []


@pytest.mark.asyncio
async def test_protected_bucket_is_skipped(hold_rt):
    """protected 桶被 hooks 两池共同排除：预脱水同样跳过。"""
    bucket_mgr, stub = hold_rt
    await hold_dispatch(_LONG_CONTENT, importance=5)
    bucket_id = (await bucket_mgr.list_all(include_archive=False))[0]["id"]
    await trace_core(bucket_id, protected=1)
    stub.dehydrate_calls.clear()

    assert await predehydrate_for_surface(bucket_id) is False
    assert stub.dehydrate_calls == []


@pytest.mark.asyncio
async def test_pinned_bucket_still_predehydrates(hold_rt):
    """pinned 桶常驻浮现池置顶（📌）：类型落在 permanent/ 也要放行。"""
    bucket_mgr, stub = hold_rt
    await hold_dispatch(_LONG_CONTENT, pinned=True)
    await _drain_background_tasks()
    stub.dehydrate_calls.clear()

    bucket = (await bucket_mgr.list_all(include_archive=False))[0]
    assert bucket["metadata"].get("pinned")
    assert await predehydrate_for_surface(bucket["id"]) is True
    assert stub.dehydrate_calls == [bucket["content"]]
