"""活化因子：记忆会自己冒出来（spontaneous recall）。

不是所有记忆都该被想起，只有「揪着心」的才拿 spark：
  - 答应的事、没做完的事、健康、重要日期、强情绪、关系变化
spark 决定每次 roll 被击中的概率；kind 决定什么时候 roll：
  hot   → 聊天时（on_llm_request）注入「你此刻突然想起…」
  cold  → 空闲时（后台 loop）主动发消息
冷却按重要性对数衰减：越重要，冷却越短（见 cooldown_days）。
每次命中 spark 衰减，问过的事不会反复问。
"""
from __future__ import annotations

import math
import random
import re
import time
from datetime import date, datetime

# (正则, kind, spark, followup, note)
_SPARK_RULES: list[tuple[str, str, float, int, str]] = [
    # 健康类：最该被主动关心
    (r"(医院|住院|生病|发烧|嗓子|身体|不舒服|体检|吃药|手术|受伤)", "cold", 0.62, 1, "身体/健康"),
    # 答应的事 / 承诺
    (r"(答应|承诺|说好|保证|一定|我会|我会一直)", "hot", 0.68, 1, "答应的事"),
    # 未完成 / 进行中
    (r"(还在|没做完|没出结果|没跑通|没补|准备|打算|下次|回头|等.{0,4}再|改天|过几天)", "hot", 0.62, 1, "未完成"),
    # 重要日期 / 事件
    (r"(生日|考试|面试|出差|旅行|机票|火车|结婚|纪念日|放假|开学)", "cold", 0.56, 1, "重要日期"),
    # 强情绪
    (r"(难过|委屈|吵架|生气|伤心|哭|崩溃|失眠|焦虑|压力大)", "cold", 0.50, 0, "情绪"),
    # 关系变化
    (r"(在一起|分手|闹翻|和好|是对象|是朋友|暧昧)", "hot", 0.55, 1, "关系"),
]


def seed_for(text: str) -> dict | None:
    """给一条记忆算活化种子；不该活化的返回 None。"""
    t = str(text or "")
    if not t:
        return None
    for pat, kind, spark, followup, note in _SPARK_RULES:
        if re.search(pat, t):
            return {"kind": kind, "spark": spark, "followup": followup, "note": note}
    return None


def cooldown_days(importance: float, *, cd_min: float = 0.5, cd_max: float = 30.0) -> float:
    """冷却天数：越重要越短，对数衰减。

    imp=0.1 → ≈21.8 天；0.5 → ≈8.2；0.9 → ≈1.7；1.0 → 0.5
    """
    imp = max(0.0, min(1.0, float(importance or 0.0)))
    t = 1.0 - math.log10(1.0 + 9.0 * imp)  # 1 → 0
    return round(max(0.0, cd_min + (cd_max - cd_min) * t), 2)


def hit_probability(
    spark: float,
    *,
    hit_count: int = 0,
    last_hit_at: float = 0.0,
    cd_days: float = 7.0,
    now: float | None = None,
) -> float:
    """当前这一轮被击中的概率（0 表示冷却中）。"""
    now_ts = time.time() if now is None else float(now)
    if last_hit_at and now_ts - float(last_hit_at) < float(cd_days) * 86400:
        return 0.0
    decay = 0.7 ** max(0, int(hit_count or 0))
    return max(0.0, min(1.0, float(spark or 0.0) * decay))


def should_hit(
    spark: float,
    *,
    hit_count: int = 0,
    last_hit_at: float = 0.0,
    cd_days: float = 7.0,
    rng: random.Random | None = None,
) -> bool:
    p = hit_probability(
        spark, hit_count=hit_count, last_hit_at=last_hit_at, cd_days=cd_days
    )
    if p <= 0:
        return False
    r = (rng or random).random()
    return r < p


# ---------- 重要日期：离日期越近，越容易被想起 ----------

_DATE_ABS = re.compile(r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*[日号]?")
_DATE_MD = re.compile(r"(?<![\d/.])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_DATE_SLASH = re.compile(r"(?<![\d/.])(\d{1,2})\s*/\s*(\d{1,2})(?![\d/.])")
_REL_DAYS = [
    ("大后天", 3),
    ("后天", 2),
    ("明天", 1),
    ("今天", 0),
    ("下周", 7),
    ("下个星期", 7),
    ("下个月", 30),
    ("这个月", 15),
]


def days_until(text: str, now: float | None = None) -> int | None:
    """从一条记忆里猜「离那天还有几天」。猜不到返回 None。

    支持：2026-11-25 / 11月25日 / 11/25 / 明天 / 后天 / 下周 / 下个月
    """
    t = str(text or "")
    if not t:
        return None
    now_ts = time.time() if now is None else float(now)
    today = datetime.fromtimestamp(now_ts).date()
    for word, d in _REL_DAYS:
        if word in t:
            return d
    m = _DATE_ABS.search(t)
    if m:
        try:
            target = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
        return (target - today).days
    m = _DATE_MD.search(t) or _DATE_SLASH.search(t)
    if m:
        mo, dd = int(m.group(1)), int(m.group(2))
        try:
            target = date(today.year, mo, dd)
        except ValueError:
            return None
        delta = (target - today).days
        if delta < -1:  # 今年已经过了，按明年算
            try:
                target = date(today.year + 1, mo, dd)
            except ValueError:
                return None
            delta = (target - today).days
        return delta
    return None


def date_proximity_factor(days: int | None) -> float:
    """离得越近，倍率越高；过了就压低。"""
    if days is None:
        return 1.0
    d = int(days)
    if d < 0:
        return 0.3
    if d == 0:
        return 2.0
    if d <= 2:
        return 1.6
    if d <= 5:
        return 1.3
    if d <= 10:
        return 1.0
    if d <= 30:
        return 0.6
    return 0.3


def proximity_factor(note: str, content: str, now: float | None = None) -> float:
    """只对「重要日期」类火花按时效加权，别的返回 1.0。"""
    if str(note or "") != "重要日期":
        return 1.0
    return date_proximity_factor(days_until(content, now))


__all__ = [
    "seed_for",
    "cooldown_days",
    "hit_probability",
    "should_hit",
    "days_until",
    "date_proximity_factor",
    "proximity_factor",
]
