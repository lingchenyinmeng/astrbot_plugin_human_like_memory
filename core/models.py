"""Human-like Memory · 数据结构与枚举。

设计要点：
- Memory  = 一次对话总结出来的一条长期记忆（粗粒度）
- Atom    = 从记忆里拆出来的原子（细粒度，带性质和时间）
- 短时库用 status 区分：candidate（暂存）→ active / dormant / dropped
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AtomStatus(str, Enum):
    """原子生命周期。

    candidate  短时区：新写入、还没定性，默认只活 48 小时
    active     正式库：定性了，按类型算 TTL
    dormant    待定：到期没定性，压最底层（不主动召回，但相似时能翻出来）
    archived   遗忘库：正式记忆放久了沉下去。**完全不参与检索**，
               只有同样的事被重新提起时才会被拉起来
    dropped    丢弃：重要性太低，压根不留
    expired    过期：正式记忆到期
    forgotten  遗忘：删除前的最后状态
    """

    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"
    DROPPED = "dropped"
    EXPIRED = "expired"
    FORGOTTEN = "forgotten"


class Layer(str, Enum):
    """谁对谁（主谓结构）。"""

    OWNER_TO_ME = "owner_to_me"
    OWNER_TO_OTHER = "owner_to_other"
    ME_TO_OWNER = "me_to_owner"
    ME_TO_OTHER = "me_to_other"
    OTHER_TO_ME = "other_to_me"
    OTHER_TO_OTHER = "other_to_other"


class MType(str, Enum):
    """话语类型。"""

    COMMAND = "command"
    PROMISE = "promise"
    PRAISE = "praise"
    BLAME = "blame"
    FACT = "fact"
    EMOTION = "emotion"
    REQUEST = "request"
    SMALLTALK = "smalltalk"


class SubType(str, Enum):
    """子类型（只有三种 type 有）。"""

    RULE = "rule"          # command: 长期规矩
    ACTION = "action"      # command: 一次性任务
    TEACHING = "teaching"  # command: 教导
    PERSON = "person"      # fact: 人物事实
    EVENT = "event"        # fact: 事件经过
    OBSERVE = "observe"    # fact: 群聊旁观
    BEHAVIOR = "behavior"  # blame: 行为批评
    IDENTITY = "identity"  # blame: 人格否定


class Nature(str, Enum):
    """话语性质（定性结果）。"""

    JOKE = "joke"
    SCOLD = "scold"
    REQUEST = "request"
    FACT = "fact"
    QUESTION = "question"
    EMOTION = "emotion"
    JOKE_GUESS = "joke_guess"  # 0.6-0.7 自己顶的「多半是玩笑」


class Privacy(str, Enum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"  # 情感/隐私，一个字不外带


# 各类型的基础 TTL（天）
BASE_TTL: dict[str, float] = {
    "episodic": 7,
    "factual": 180,
    "relational": 90,
    "preference": 60,
    "planned": 2,
    "unknown": 30,
}


@dataclass(slots=True)
class Memory:
    """一条长期记忆（对话总结的产物）。"""

    content: str = ""
    summary: str = ""
    personality_summary: str = ""  # 注入用的第一人称版本（可选）
    key_facts: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    participant_identities: list[dict] = field(default_factory=list)
    sentiment: str = "neutral"
    importance: float = 0.5
    session_id: str = ""
    persona_id: str = ""
    source_session_id: str = ""   # 原始会话（形如 平台名:FriendMessage:123456789）
    is_group: bool = False
    create_time: float = field(default_factory=time.time)
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: int = 0


@dataclass(slots=True)
class Atom:
    """一条记忆原子（细粒度，带性质与生命周期）。"""

    memory_id: int = 0
    content: str = ""

    # 分类
    layer: str = ""
    mtype: str = ""
    subtype: str = ""
    classify_conf: float = 0.0

    # 定性
    nature: str = ""
    nature_conf: float = 0.0
    nature_source: str = ""  # rule / history / assume / owner
    # 判据：为什么判成这个性质。空着也行，但填了就能复核。
    nature_reason: str = ""
    # 原文摘录或触发的关键词，方便回看当时看到的是什么
    nature_evidence: str = ""
    # 谁判的：rule（正则）/ llm（模型）/ owner（主人纠正）/ history（历史参考）
    nature_by: str = ""
    # 改判次数，改一次 +1，旧值进 atom_nature_history
    nature_ver: int = 0

    # 来源与上下文
    who_id: str = ""
    who_name: str = ""
    # 这条事件牵扯到哪些人（人库里的 person_id，不写名字）
    # 一条事件可以同时挂好几个人；反查时靠这些 id 跳进人表
    person_ids: list[str] = field(default_factory=list)
    said_at: float = 0.0
    context: list[dict] = field(default_factory=list)
    source: str = ""          # group / private
    with_owner: bool = False  # 是不是与主人的私聊
    partition_key: str = ""
    topic: str = ""
    privacy: str = ""
    # 证据分层：这条记忆是「怎么知道的」（亲历/查到/听说/推测/想象）
    evidence: str = ""

    # 生命周期
    status: str = AtomStatus.CANDIDATE.value
    importance: float = 0.5
    create_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    last_reinforce: float = 0.0
    reinforce_count: int = 0
    ttl_days: float = 2.0
    expires_at: float = 0.0
    # 被检索命中过几次。`touch()` 会 +1 并顺手续命——
    # 被反复想起的事不该跟没被想起的一样按时死掉。
    recall_count: int = 0
    # 情绪二维（Russell 环形模型）：效价 -1~1、唤醒度 0~1。
    # nature 说的是「这是什么话」，这两个说的是「这句话什么温度」。
    valence: float = 0.0
    arousal: float = 0.0

    session_id: str = ""
    persona_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    atom_id: int = 0

    def compute_ttl(self) -> float:
        """按类型算存活天数。"""
        base = BASE_TTL.get(self.mtype or "unknown", 30)
        imp_factor = 0.5 + max(0.0, min(1.0, self.importance))
        reinforce_factor = 1.0 + min(0.5, self.reinforce_count * 0.1)
        # 证据越弱，活得越短——「我猜的」不该跟「亲历的」一样久
        ev_factor = 0.5 + 0.5 * evidence_weight(getattr(self, "evidence", ""))
        return max(1.0, base * imp_factor * reinforce_factor * ev_factor)


# 证据分层：这条记忆是「怎么知道的」。
# 借鉴 Qingyi 的思路——亲历/听说/查到/推测/想象 永远不混写成事实。
# 人格归纳时按这里的权重降权，避免把猜的当成真的。
EVIDENCE_TYPES = ("亲历", "查到", "听说", "推测", "想象")
EVIDENCE_WEIGHT = {
    "亲历": 1.0,   # 她自己做的/看到的
    "查到": 0.8,   # 翻文件、查日志、搜出来的
    "听说": 0.7,   # 别人说的
    "推测": 0.4,   # 她自己猜的
    "想象": 0.2,   # 假设、假如、梦
}
EVIDENCE_CN = {"亲历": "亲历", "查到": "查到", "听说": "听说",
               "推测": "推测", "想象": "想象"}


def evidence_weight(kind: str) -> float:
    return EVIDENCE_WEIGHT.get(str(kind or ""), 0.6)


__all__ = [
    "EVIDENCE_TYPES", "EVIDENCE_WEIGHT", "EVIDENCE_CN", "evidence_weight",
    "AtomStatus", "Layer", "MType", "SubType", "Nature", "Privacy",
    "BASE_TTL", "Memory", "Atom",
]
