"""短期未完成线索：刚才那句话没说完、问了没答、玩笑没接上。

和 `gestalt.unfinished`（长期待办）的区别：
- **那个**是「欠着的那首歌」「插件没做完」，能挂几个月
- **这个**是「三分钟前问了他一句话没回」，过了就凉

借鉴 self_evolution 的 UnfinishedCue：TTL 180 秒。
为什么 180：超过三分钟，人自己也会转移注意力——留着只会让她反复翻旧账。

表放在 memory 库（和 atoms 同一个），按会话（umo）隔离。
"""
from __future__ import annotations

import time

import aiosqlite

TTL_SECONDS = 180.0

KIND_QUESTION = "unanswered_question"   # 她问了，对方没答
KIND_JOKE = "unfinished_joke"           # 她抛了梗，没人接
KIND_TOPIC = "unfinished_topic"         # 话题被岔开

KIND_CN = {
    KIND_QUESTION: "问了没答",
    KIND_JOKE: "玩笑没接上",
    KIND_TOPIC: "话被打断",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS unfinished_cues (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scope      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    content    TEXT DEFAULT '',
    atom_id    INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cue_scope ON unfinished_cues(scope, expires_at);
"""


class CueStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def add(
        self,
        scope: str,
        kind: str,
        content: str,
        *,
        atom_id: int = 0,
        ttl: float = TTL_SECONDS,
    ) -> int:
        """记一条未完成线索。同 scope 同 kind 同内容的不重复记。"""
        now = time.time()
        text = str(content or "").strip()[:80]
        if not text:
            return 0
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id FROM unfinished_cues WHERE scope = ? AND kind = ? "
                "AND content = ? AND expires_at > ? LIMIT 1",
                (scope, kind, text, now),
            )
            if await cur.fetchone():
                return 0
            cur = await db.execute(
                "INSERT INTO unfinished_cues(scope, kind, content, atom_id, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (scope, kind, text, int(atom_id or 0), now, now + max(30.0, ttl)),
            )
            await db.commit()
            return int(cur.lastrowid)

    async def active(self, scope: str, limit: int = 5) -> list[dict]:
        """还没过期的线索，新的在前。"""
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT kind, content, atom_id, created_at, expires_at "
                "FROM unfinished_cues WHERE scope = ? AND expires_at > ? "
                "ORDER BY created_at DESC LIMIT ?",
                (scope, now, max(1, limit)),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def resolve(self, scope: str, kind: str = "") -> int:
        """标记已接上（删掉，不是留着看）。"""
        async with aiosqlite.connect(self.db_path) as db:
            if kind:
                cur = await db.execute(
                    "DELETE FROM unfinished_cues WHERE scope = ? AND kind = ?",
                    (scope, kind),
                )
            else:
                cur = await db.execute(
                    "DELETE FROM unfinished_cues WHERE scope = ?", (scope,)
                )
            await db.commit()
            return cur.rowcount

    async def cleanup(self) -> int:
        """清过期的。维护循环调用。"""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM unfinished_cues WHERE expires_at <= ?", (time.time(),)
            )
            await db.commit()
            return cur.rowcount

    async def stats(self) -> dict:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            async def one(sql: str, params=()) -> int:
                cur = await db.execute(sql, params)
                return int((await cur.fetchone())[0])

            return {
                "live": await one(
                    "SELECT COUNT(*) FROM unfinished_cues WHERE expires_at > ?", (now,)
                ),
                "stale": await one(
                    "SELECT COUNT(*) FROM unfinished_cues WHERE expires_at <= ?", (now,)
                ),
            }


def format_cues(cues: list[dict]) -> str:
    """把未完成线索写成一两句人话，给注入用。"""
    if not cues:
        return ""
    bits = []
    for c in cues[:3]:
        kind = str(c.get("kind") or "")
        content = str(c.get("content") or "").strip()
        if not content:
            continue
        label = KIND_CN.get(kind, "")
        bits.append(f"{label}：「{content}」" if label else f"「{content}」")
    if not bits:
        return ""
    return "刚才还有没说完的——" + "；".join(bits) + "。"


__all__ = [
    "CueStore",
    "TTL_SECONDS",
    "KIND_QUESTION",
    "KIND_JOKE",
    "KIND_TOPIC",
    "KIND_CN",
    "format_cues",
]
