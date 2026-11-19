"""Human-like Memory · 分类（谁对谁 + 什么类型 + 子类型）与上下文提取。

输入是 key_facts（LLM 从对话里提炼的短句），输出是带标签的 Atom 雏形。
分类全部走规则，不调模型。
"""
from __future__ import annotations

import re
import time
from typing import Any

from .models import Atom, AtomStatus, MType
from .nature import apply_history, classify_nature, resolve_nature, to_tier

# ---------- 关键词表 ----------

_CMD = r"去|帮我|给我|马上|现在|立刻|不许|不准|别|以后|每次|记得|要一直|改成|教|学会|要求|让我|叫我|喊我|吩咐|叮嘱"
_PROMISE = r"答应|承诺|约定|说好|一定|我会|保证"
_PRAISE = r"可爱|厉害|聪明|好看|不错|挺好|棒|优秀|喜欢|真行|靠谱|夸"
_BLAME = r"蠢|笨|傻|烦|啰嗦|话多|点偏|错了|不行|没用|废物|滚|嫌|骂"
_EMOTION = r"累|困|难过|开心|烦|委屈|生气|害怕|想哭|难受|高兴|崩溃|爽"
_REQUEST = r"能不能|可不可以|帮我|请问|怎么办|要不要|吗[？?]?$|呢[？?]?$|为什么|是不是"
_SMALLTALK = r"^哈|^嗯|^哦|^在吗|^早|^晚|吃了|睡了|天气|晚安"

# command 细分
_RULE_WORDS = r"以后|每次|永远|不许|不准|别再|要一直|从现在起|记住|规矩|定个规矩"
_TEACH_WORDS = r"教你|应该|改成|别用|学会|这样说|这样说才对|下次这么|教你一句"
_ACTION_WORDS = r"去|帮我|给我|现在|赶紧|马上"

# blame 细分
_IDENTITY_WORDS = r"傻|蠢|笨|废物|没用|垃圾|滚|你是鱼|智障"

# 隐私关键词（情感/隐私类 → sensitive）
_SENSITIVE = r"难过|委屈|哭|分手|吵架|家里|身体|病|钱|工资|工资|秘密|不想说|心事|失眠|焦虑|压力"


def _has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, str(text or "")))


def classify_type(text: str) -> tuple[str, float]:
    """判 type（什么性质的话）。"""
    t = str(text or "")
    if _has(_PROMISE, t):
        return MType.PROMISE.value, 0.78
    if _has(_PRAISE, t):
        return MType.PRAISE.value, 0.75
    if _has(_BLAME, t):
        return MType.BLAME.value, 0.76
    if _has(_CMD, t):
        return MType.COMMAND.value, 0.74
    if _has(_EMOTION, t):
        return MType.EMOTION.value, 0.72
    if _has(_REQUEST, t):
        return MType.REQUEST.value, 0.70
    if _has(_SMALLTALK, t):
        return MType.SMALLTALK.value, 0.65
    return MType.FACT.value, 0.60


def classify_subtype(mtype: str, text: str) -> str:
    """只给 command / fact / blame 三种细分。"""
    t = str(text or "")
    if mtype == MType.COMMAND.value:
        if _has(_RULE_WORDS, t):
            return "rule"
        if _has(_TEACH_WORDS, t):
            return "teaching"
        return "action"
    if mtype == MType.BLAME.value:
        return "identity" if _has(_IDENTITY_WORDS, t) else "behavior"
    if mtype == MType.FACT.value:
        # 有人称 + 描述 → person；纯事件 → event
        if _has(r"是|叫|喜欢|住在|年龄|生日|职业|工作", t) and _has(r"[一-龥]{2,4}", t):
            return "person"
        return "event"
    return ""


def classify_layer(
    text: str,
    *,
    owner_id: str = "",
    owner_name: str = "",
    who_id: str = "",
    who_name: str = "",
    source: str = "",
    identities: list[dict] | None = None,
    bot_names: list[str] | None = None,
) -> str:
    """判 layer（谁对谁）。

    之前两个坑（2026-11-19 修）：
    1. 只认「铃光/我」当目标，别人说主人的事（other→owner）没有这一档，全掉进 other_to_other。
    2. 名字只在句首（startswith）才算提到，名字在中间 / 用代称就漏了，群聊默认 other_to_other。
    现在：整句找「不是说话人的第一个名字」当目标，补上 other_to_owner。
    """
    owner_name = str(owner_name or "").strip()
    t = str(text or "")
    owner = str(owner_id or "").strip()
    who = str(who_id or "").strip()
    is_owner = bool(owner and who == owner) or bool(
        owner_name and who_name and owner_name in who_name
    )
    bots = {str(x).strip() for x in (bot_names or []) if str(x).strip()} | {"铃光", "我", "自己"}

    # 名字 → (person_id, 是不是 bot)
    name_to_pid: dict[str, tuple[str, bool]] = {}
    for ident in identities or []:
        if not isinstance(ident, dict):
            continue
        sid = str(ident.get("sender_id") or "").strip()
        if not sid:
            continue
        isb = bool(ident.get("is_bot"))
        for key in ("display_name", *(ident.get("aliases") or [])):
            k = str(key or "").strip()
            if k:
                name_to_pid[k] = (sid, isb)
    bot_ids = {sid for sid, isb in name_to_pid.values() if isb}
    who_is_bot = bool(who and who in bot_ids) or bool(
        who_name and who_name.strip() in bots
    )

    # 目标：句子里出现的、不是说话人本人的第一个名字
    target_sid, target_is_bot = "", False
    for name, (sid, isb) in name_to_pid.items():
        if not name or name not in t:
            continue
        if sid and who and sid == who:
            continue  # 名字就是说话人，不是目标
        target_sid, target_is_bot = sid, isb
        break

    owner_mentioned = bool(owner_name and owner_name in t and not is_owner)
    bot_mentioned = bool(any(b and b in t for b in bots) and not who_is_bot)
    target_is_owner = bool(owner and target_sid and target_sid == owner) or owner_mentioned
    target_is_me = target_is_bot or bot_mentioned

    if who_is_bot:
        return "me_to_owner" if target_is_owner else "me_to_other"
    if is_owner:
        if target_is_me:
            return "owner_to_me"
        if target_is_owner:
            return "owner_to_other"
        # 点到具体「别人」→ 对别人；什么都没点 → 默认对铃光
        if target_sid:
            return "owner_to_other"
        return "owner_to_me"
    # 说话人是别人
    if target_is_owner:
        return "other_to_owner"
    if target_is_me:
        return "other_to_me"
    if source == "group":
        return "other_to_other"
    return "other_to_me"


def infer_fact_speaker(
    fact: str,
    *,
    name_map: dict[str, tuple[str, str]] | None = None,
    default: tuple[str, str] = ("", ""),
    bot_id: str = "",
    bot_name: str = "",
) -> tuple[str, str]:
    """从一条 key_fact 的措辞推出「这话是谁说的」。

    总结器被要求把事实写成「噩壤说…」「鲸宝说他…」这种，所以：
    - 以「我」或 bot 名字开头 → 铃光自己说的
    - 以某个已知名字开头 → 那个人说的（取最长匹配，避免「噩」吃掉「噩壤」）
    - 都推不出 → 用这段对话的第一发言人（default）
    """
    t = str(fact or "").strip()
    if not t:
        return default
    bname = str(bot_name or "").strip()
    if t.startswith("我") or (bname and t.startswith(bname)):
        return (bot_id, bname or "我")
    best: tuple[str, str, str] | None = None
    for nm, (sid, real_name) in (name_map or {}).items():
        if nm and t.startswith(nm):
            if best is None or len(nm) > len(best[0]):
                best = (nm, sid, real_name)
    if best:
        return (best[1], best[2])
    return default


# 重要性基线：按「这是什么类型的话」给一个底线值。
# LLM 总结时经常随口给 0.3（它的兜底默认），导致 42% 的原子全卡在 0.3，
# 区分度起不来。所以改成「基线为主、LLM 只做微调」。
IMPORTANCE_BASE: dict[str, float] = {
    "command": 0.70,    # 规矩、要求——她得记住
    "promise": 0.75,    # 答应的事——最该记
    "praise": 0.50,     # 夸一句，记着但没那么重
    "blame": 0.60,      # 批评——得知道哪里错了
    "fact": 0.50,       # 陈述
    "emotion": 0.55,    # 情绪——比事实重一点
    "request": 0.45,    # 提问
    "smalltalk": 0.30,  # 闲聊
}


def importance_for(mtype: str, llm_value) -> float:
    """算重要性：类型基线为主，LLM 给的值只做微调。

    LLM 经常返回 0.3 这种兜底值（占全库 42%），不能让它主导。
    策略：LLM 值 ≤ 0.35 视为「没给」，直接用基线；否则取基线和它的平均。
    """
    base = IMPORTANCE_BASE.get(str(mtype or ""), 0.5)
    try:
        v = float(llm_value or 0)
    except (TypeError, ValueError):
        return base
    if v <= 0.35:
        return base
    return round((base + min(1.0, v)) / 2, 2)


# 证据分层判定：这条记忆是「怎么知道的」。
# 借鉴 Qingyi——亲历/听说/查到/推测/想象 永远不混写成"事实"。
_EVIDENCE_PATTERNS = (
    ("想象", r"(假如|如果.{0,6}会|假设|想象|梦见|要是)"),
    ("推测", r"(可能|大概|也许|估计|应该是|我猜|觉得.{0,4}会)"),
    ("查到", r"(查了|查到|搜了|搜到|翻了?.{0,2}(日志|文件|代码|记录)|看代码|文档|README)"),
)


def classify_evidence(
    text: str,
    *,
    layer: str = "",
    who_id: str = "",
    owner_id: str = "",
    source: str = "",
) -> str:
    """判这条记忆是亲历/查到/听说/推测/想象。

    顺序很重要：先排除「想象/推测」，再认「查到」，最后按说话人定亲历/听说。
    因为「他可能知道」这种句子既像听说又像推测，推测更准确。
    """
    t = str(text or "")
    for name, pat in _EVIDENCE_PATTERNS:
        if re.search(pat, t):
            return name
    # 她自己做的事 → 亲历。注意：主人说的事也是「她听来的」，
    # 不能因为 who 是主人就算亲历——那是把别人的话记成自己的经历。
    if str(layer or "").startswith("me_"):
        return "亲历"
    return "听说"


# 情绪二维（Russell 环形模型）：效价 + 唤醒度。
# nature 说的是「这是什么话」，valence/arousal 说的是「这句话什么温度」。
_NATURE_AFFECT: dict[str, tuple[float, float]] = {
    # nature: (valence 效价 -1~1, arousal 唤醒 0~1)
    "praise": (0.75, 0.55),
    "joke": (0.55, 0.65),
    "emotion": (0.20, 0.70),
    "fact": (0.05, 0.30),
    "request": (0.00, 0.40),
    "scold": (-0.70, 0.80),
}


def classify_affect(nature: str, text: str) -> tuple[float, float]:
    """返回 (valence, arousal)。"""
    v, a = _NATURE_AFFECT.get(str(nature or ""), (0.0, 0.35))
    t = str(text or "")
    # 强情感词微调
    if re.search(r"(太好了|最喜欢|开心|高兴|谢谢)", t):
        v = min(1.0, v + 0.2)
        a = min(1.0, a + 0.1)
    if re.search(r"(难过|难受|委屈|想哭|对不起)", t):
        v = max(-1.0, v - 0.25)
        a = min(1.0, a + 0.1)
    if re.search(r"[!！]{1,}", t):
        a = min(1.0, a + 0.15)
    return round(v, 2), round(a, 2)


def is_sensitive(text: str, mtype: str, nature: str) -> bool:
    """情感/隐私类内容 → 打 sensitive 标签（红线第二档靠它）。"""
    if mtype == MType.EMOTION.value or nature == "emotion":
        return True
    return _has(_SENSITIVE, str(text or ""))


def extract_context(messages: list, count: int = 2, max_chars: int = 100) -> list[dict]:
    """取最近 N 条消息当上文。"""
    out: list[dict] = []
    for msg in list(messages)[-max(1, count):]:
        try:
            text = str(getattr(msg, "content", "") or "").strip()
        except Exception:
            text = ""
        if not text:
            continue
        try:
            name = str(getattr(msg, "sender_name", "") or getattr(msg, "sender_id", "") or "")
        except Exception:
            name = ""
        out.append({"who": name, "text": text[:max_chars]})
    return out


def build_atoms(
    key_facts: list[str],
    *,
    memory_id: int = 0,
    owner_id: str = "",
    owner_name: str = "",
    who_id: str = "",
    who_name: str = "",
    source: str = "",
    said_at: float = 0.0,
    context: list[dict] | None = None,
    identities: list[dict] | None = None,
    history: list[dict] | None = None,
    importance: float = 0.5,
    session_id: str = "",
    persona_id: str = "",
    topic: str = "",
    fact_who: dict[str, tuple[str, str]] | None = None,
    bot_names: list[str] | None = None,
) -> list[Atom]:
    """把 key_facts 变成带完整标签的原子列表。

    fact_who: 每条 fact → (who_id, who_name)，逐条归属说话人；
    没给的 fact 用默认 who_id/who_name（这段对话的第一发言人）。
    """
    atoms: list[Atom] = []
    now = time.time()
    for fact in key_facts:
        fact = str(fact or "").strip()
        if not fact:
            continue

        fw = (fact_who or {}).get(fact) or (who_id, who_name)
        f_who_id, f_who_name = str(fw[0] or ""), str(fw[1] or "")

        mtype, tconf = classify_type(fact)
        subtype = classify_subtype(mtype, fact)
        layer = classify_layer(
            fact, owner_id=owner_id, owner_name=owner_name, who_id=f_who_id,
            who_name=f_who_name, source=source, identities=identities,
            bot_names=bot_names,
        )

        # 定性：按 mtype 定基调，正则只细化语气（避免两套规则打架）
        raw_nature, raw_conf, raw_reason, raw_evidence = resolve_nature(mtype, fact)
        nature, nconf, nsrc = apply_history(raw_nature, raw_conf, history or [])
        final_nature, tier_src, need_ask = to_tier(nature, nconf)
        if nsrc:
            tier_src = nsrc

        privacy = "sensitive" if is_sensitive(fact, mtype, final_nature) else "normal"
        with_owner = bool(
            owner_id and f_who_id == owner_id and source == "private"
        )

        _affect = classify_affect(final_nature, fact)
        atom = Atom(
            memory_id=memory_id,
            content=fact,
            layer=layer,
            mtype=mtype,
            subtype=subtype,
            classify_conf=tconf,
            nature=final_nature,
            nature_conf=nconf,
            nature_source=tier_src,
            # 判据：为什么判成这个性质、看到的是哪几个字、谁判的
            nature_reason=raw_reason,
            nature_evidence=raw_evidence,
            nature_by=tier_src or "rule",
            nature_ver=0,
            who_id=f_who_id,
            who_name=f_who_name,
            said_at=float(said_at or now),
            context=list(context or []),
            source=source,
            with_owner=with_owner,
            topic=topic,
            evidence=classify_evidence(
                fact, layer=layer, who_id=f_who_id,
                owner_id=owner_id, source=source,
            ),
            valence=_affect[0],
            arousal=_affect[1],
            privacy=privacy,
            status=AtomStatus.CANDIDATE.value,
            importance=importance_for(mtype, importance),
            create_time=now,
            last_access=now,
            session_id=session_id,
            persona_id=persona_id,
            metadata={},
        )
        if need_ask:
            atom.metadata["ask_owner"] = True
            atom.metadata["nature_guess"] = raw_nature
            atom.metadata["nature_conf"] = round(raw_conf, 2)
        atoms.append(atom)
    return atoms


__all__ = [
    "classify_type", "classify_subtype", "classify_layer",
    "is_sensitive", "extract_context", "build_atoms", "infer_fact_speaker",
]
