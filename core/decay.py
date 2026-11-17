"""遗忘曲线：比「到期就 archived」更接近人脑。

原来的 TTL 是一刀切：到日子就死，不管这事多重要、被想起过几次。
问题是——
  1. 重要的事该活得久，现在跟琐事一样死
  2. 被反复想起的事应该「回血」，现在 `reinforce_count` 只加数字不续命

公式（借鉴 memory_system 的 EDM，加了回忆奖励）：

    strength = (importance ** 0.8) * exp(-0.05 * hours / max(0.3, importance))
               + recall_bonus

- `importance ** 0.8`：重要度对衰减的影响不是线性的，0.9 比 0.3 活得明显久
- `exp(-0.05 * hours / importance)`：重要的事衰减更慢（除数大）
- `recall_bonus`：每被想起一次 +0.08，封顶 0.4——想起五次以上就基本不再衰减
"""
from __future__ import annotations

import math

# 基础衰减率（每小时）。0.05 对应半衰期约 14 小时——太快的用 importance 除回去。
BASE_DECAY = 0.05
# 每次被回忆回多少
RECALL_BONUS = 0.08
# 回忆回血上限
RECALL_CAP = 0.40


def strength(
    *,
    importance: float,
    hours: float,
    recalls: int = 0,
    reinforce: int = 0,
) -> float:
    """算这条记忆现在的「强度」。0 到 1 左右。

    :param importance: 0-1
    :param hours: 距上次访问多少小时
    :param recalls: 被检索命中过几次
    :param reinforce: 被重复说过几次（reinforce_count）
    """
    imp = max(0.05, min(1.0, float(importance or 0.5)))
    h = max(0.0, float(hours or 0))
    r = max(0, int(recalls or 0))
    rf = max(0, int(reinforce or 0))

    # 重要度当除数：越重要，衰减越慢
    divisor = max(0.30, imp)
    core = (imp ** 0.8) * math.exp(-BASE_DECAY * h / divisor)

    bonus = min(RECALL_CAP, RECALL_BONUS * (r + 0.5 * rf))
    return max(0.0, min(1.2, core + bonus))


def should_decay(
    *,
    importance: float,
    hours: float,
    recalls: int = 0,
    reinforce: int = 0,
    floor: float = 0.15,
) -> bool:
    """强度掉到底线以下就该进遗忘库了。"""
    return strength(
        importance=importance, hours=hours, recalls=recalls, reinforce=reinforce
    ) < floor


def remember_bonus(atom_like) -> float:
    """给一个带 importance/recalls/reinforce 的对象算当前权重（用于排序）。"""
    return strength(
        importance=float(getattr(atom_like, "importance", 0.5) or 0.5),
        hours=0.0,   # 排序时不算时间衰减，交给调用方
        recalls=int(getattr(atom_like, "recall_count", 0) or 0),
        reinforce=int(getattr(atom_like, "reinforce_count", 0) or 0),
    )


__all__ = ["strength", "should_decay", "remember_bonus", "BASE_DECAY",
           "RECALL_BONUS", "RECALL_CAP"]
