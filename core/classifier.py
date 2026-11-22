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

# ---------- 关键词表（带权重 + 上下文约束）----------
# 旧版是「一个词命中就定」，别/去/现在/喜欢/不行 这种日常高频词会把大量陈述带偏。
# 现在改成打分制：词给分，上下文约束单独给分，取最高分；
# 没有上下文支撑的通用词单独出现拿不到分，落到 fact。

# (mtype, 权重, 正则)
_TYPE_RULES: list[tuple[str, float, str]] = [
    # 承诺：我答应/保证/说好 + 对象
    ("promise", 1.0, r"(?:我|铃光|小光)[^。！？]{0,10}(?:答应|承诺|说好|保证|一定|会)(?:你|他|她|以后|要)"),
    ("promise", 0.8, r"(?:答应你|答应过你|保证不|承诺过|说好不)"),
    # 夸：对象得是铃光/你/她 + 夸词（不能只是「喜欢某物」）
    ("praise", 1.0, r"(?:你|铃光|小光|她)[^。！？]{0,10}(?:真|太|挺|好|很|最)*(?:可爱|厉害|聪明|好看|棒|优秀|靠谱|乖|真好|真行)"),
    ("praise", 0.7, r"(?:夸|表扬|称赞)(?:你|我|铃光|小光)"),
    # 骂：对象得是人 + 骂词
    ("blame", 1.0, r"(?:你|铃光|小光|她)[^。！？]{0,10}(?:蠢|笨|傻|废物|没用|垃圾|白痴|智障|啰嗦|烦人)"),
    ("blame", 0.75, r"(?:凶你|凶我|骂你|骂我|凶她|骂她)"),
    # 命令：祈使 / 第二人称规矩 / 让我 / 要求 / 教我
    ("command", 1.0, r"(?:让我|要求我|吩咐我|叮嘱我|让我以后|要我)[^。！？]{0,50}"),
    ("command", 0.85, r"(?:要求|规定|吩咐|叮嘱)[^。！？]{0,40}"),
    ("command", 0.8, r"(?:教|指导)我[^。！？]{0,40}"),
    ("command", 0.9, r"(?:你|铃光|小光)[^。！？]{0,8}(?:以后|每次|记住|别再|不许|不准|必须|要一直|从现在起)"),
    ("command", 0.8, r"(?:以后|每次)[^。！？]{0,24}(?:别|不许|不准|别再|记得|要一直|必须)"),
    ("command", 0.75, r"^(?:别|不许|不准|不要|去|赶紧|马上)[^。！？]{1,40}"),
    ("command", 0.6, r"(?:记得|别忘了)(?:吃|睡|回|叫|说|带|买)"),
    # 情绪：第一人称
    ("emotion", 1.0, r"(?:^我|我有点|我好|我今天|我最近|今天|最近)[^。！？]{0,8}(?:累|困|难过|开心|烦|委屈|生气|害怕|崩溃|想哭|高兴|爽)"),
    ("emotion", 0.6, r"(?:心累|烦死|气死|累死|开心死了|好难过)"),
    # 提问（允许句末带引号/括号）
    ("request", 1.0, r"[？?][」』）)\s]*$|(?:吗|呢)[。！!…」』）)\s]*$"),
    ("request", 0.8, r"(?:怎么|为什么|是不是|能不能|可不可以|要不要|请问|怎么办|有没有|会不会|在哪|几点)"),
    # 闲聊
    ("smalltalk", 1.0, r"^(?:哈+|嗯+|哦+|在吗|早|晚安|早安|晚上好|吃了|睡了)[。！!…]*$"),
    ("smalltalk", 0.5, r"(?:晚安|早安|道晚安|说睡了|睡了)"),
]

_TYPE_ORDER = ["promise", "blame", "praise", "command", "emotion", "request", "smalltalk"]

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
    """判 type（什么性质的话）。打分制：命中越多越强分越高；通用词单点出现不给分。"""
    t = str(text or "").strip()
    if not t:
        return MType.FACT.value, 0.5
    scores: dict[str, float] = {}
    for mtype, weight, pattern in _TYPE_RULES:
        if _has(pattern, t):
            scores[mtype] = scores.get(mtype, 0.0) + weight
    if not scores:
        return MType.FACT.value, 0.60
    best = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            _TYPE_ORDER.index(kv[0]) if kv[0] in _TYPE_ORDER else 99,
        ),
    )[0]
    mtype, score = best
    conf = min(0.95, 0.55 + 0.12 * score)
    return mtype, round(conf, 2)


def classify_subtype(mtype: str, text: str, *, layer: str = "") -> str:
    """细分。command: rule/action/teaching；blame: identity/behavior；
    fact: trait/relation/state/knowledge/event。
    """
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
        # 属性：句首像人名 + 属性词（"鲸宝喜欢…"「噩壤的生日是…」）
        if _has(r"^(?:[一-龥A-Za-z0-9_~ ]{2,12})(?:是|叫|喜欢|讨厌|擅长|生日|年龄|职业|住|外号)", t):
            return "trait"
        # 关系：谁和谁
        if _has(r"(?:和|跟|与)[^。！？]{0,10}(?:在一起|分手|吵架|闹翻|和好|是朋友|是对象|恋人|夫妻|暧昧)", t):
            return "relation"
        # 知识/规则
        if _has(r"(?:意思|解释|定义|叫做|指的是|规则|原理|用法|名词)", t):
            return "knowledge"
        # 状态：在干嘛、进度
        if _has(r"(?:正在|还在|目前|最近|准备|打算|要去|已经|进度|在忙)", t):
            return "state"
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


# 子类型对重要性的微调：关系 > 属性 > 知识 > 事件 > 状态
_SUBTYPE_IMPORTANCE: dict[str, float] = {
    "rule": 0.05,
    "teaching": 0.03,
    "action": 0.0,
    "trait": 0.05,
    "relation": 0.08,
    "state": -0.05,
    "knowledge": 0.03,
    "event": -0.03,
    "identity": 0.03,
    "behavior": 0.0,
}


def importance_for(
    mtype: str,
    llm_value,
    *,
    text: str = "",
    layer: str = "",
    subtype: str = "",
) -> float:
    """算重要性：类型基线 + 子类型微调 + 上下文加减，LLM 值只做微调。

    旧版「基线 + LLM 平均」把差异抹平，全库 78% 挤在 0.4–0.6。
    现在按「涉不涉及她/主人、语气强不强、信息量、子类型」加减，拉开区分度。
    """
    base = IMPORTANCE_BASE.get(str(mtype or ""), 0.5)
    base += _SUBTYPE_IMPORTANCE.get(str(subtype or ""), 0.0)
    score = base
    t = str(text or "")
    # 涉及铃光 / 主人 → 更重
    if re.search(r"(?:铃光|小光|主人|噩壤|你)", t):
        score += 0.06
    # 强语气 → 更重
    if re.search(r"[!！]{2,}|(?:太|真的|一定|必须|记住|千万)", t):
        score += 0.05
    # 主人直接对她说 / 她对主人说 → 更重
    if layer in ("owner_to_me", "me_to_owner"):
        score += 0.04
    # 信息量：长句更重，超短句更轻
    L = len(t)
    if L >= 40:
        score += 0.04
    elif L <= 10:
        score -= 0.04
    if mtype == "smalltalk" or L <= 6:
        score -= 0.06
    # LLM 值只微调（≤0.35 视为没给）
    try:
        v = float(llm_value or 0)
    except (TypeError, ValueError):
        v = 0.0
    if v > 0.35:
        score = score * 0.7 + min(1.0, v) * 0.3
    return round(max(0.1, min(1.0, score)), 2)


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
    """判这条记忆是怎么知道的：亲历 / 查到 / 听说 / 旁听 / 转述 / 推测 / 想象。

    顺序很重要：先排除「想象/推测/查到」（语气词最准），再按 layer 定来源。
    以前只要不是她自己说的就一律「听说」，90% 的库里全是「听说」，
    这一档等于废了——现在把「旁听」（群里别人之间）和「转述」（A 说 B 说）
    拆出来，权重才降得动。
    """
    t = str(text or "")
    for name, pat in _EVIDENCE_PATTERNS:
        if re.search(pat, t):
            return name
    # 她自己做的/说的 → 亲历
    if str(layer or "").startswith("me_"):
        return "亲历"
    # 转述：A 说 B 说…（两层转述）
    if re.search(r"(?:说|问|提到|讲|转述)[^。！？]{0,12}(?:说|问|提到|讲)", t):
        return "转述"
    # 群里别人之间在聊 → 旁听
    if str(layer or "") == "other_to_other":
        return "旁听"
    # 别人直接对她说 → 听说
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
        layer = classify_layer(
            fact, owner_id=owner_id, owner_name=owner_name, who_id=f_who_id,
            who_name=f_who_name, source=source, identities=identities,
            bot_names=bot_names,
        )
        subtype = classify_subtype(mtype, fact, layer=layer)

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
            importance=importance_for(mtype, importance, text=fact, layer=layer, subtype=subtype),
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
