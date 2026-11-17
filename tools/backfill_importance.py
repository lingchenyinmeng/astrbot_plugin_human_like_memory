"""回填 importance：把卡在 0.3 的按 mtype 基线重算。

为什么只改 0.3 的：LLM 给的其他值（0.5/0.7/0.9）是有意义的判断，
0.3 是它的兜底默认值，占了 42%，没区分度。

用法：
    python tools/backfill_importance.py --dry-run
    python tools/backfill_importance.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PLUGIN_ROOT))
sys.path.insert(0, PLUGIN_ROOT)

import aiosqlite

from astrbot_plugin_human_like_memory.core.classifier import (  # noqa: E402
    importance_for,
)
from astrbot_plugin_human_like_memory.tools.migrate_from_livingmemory import (  # noqa: E402
    find_data_dir,
)


def default_db() -> str | None:
    data = find_data_dir()
    if not data:
        return None
    return os.path.join(
        data, "plugin_data", "astrbot_plugin_human_like_memory", "memory.db"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="按类型基线回填重要性")
    ap.add_argument("--db", default=default_db())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.db or not os.path.isfile(args.db):
        print(f"[补重要性] 找不到记忆库: {args.db}")
        raise SystemExit(2)

    db = await aiosqlite.connect(args.db)
    db.row_factory = aiosqlite.Row
    try:
        cur = await db.execute(
            "SELECT id, mtype, importance FROM atoms "
            "WHERE ABS(importance - 0.3) < 0.001 ORDER BY id"
        )
        rows = await cur.fetchall()
        print(f"[补重要性] 卡在 0.3 的 {len(rows)} 条")

        changed = 0
        dist: dict[str, int] = {}
        t0 = time.time()
        for r in rows:
            mtype = str(r["mtype"] or "")
            new = importance_for(mtype, 0.3)
            dist[mtype] = dist.get(mtype, 0) + 1
            if new == 0.3:
                continue
            changed += 1
            if args.dry_run:
                continue
            await db.execute(
                "UPDATE atoms SET importance = ? WHERE id = ?",
                (new, int(r["id"])),
            )
            if changed % 300 == 0:
                await db.commit()
                print(f"  {changed}/{len(rows)} 用时 {time.time()-t0:.0f}s", flush=True)
        if not args.dry_run:
            await db.commit()
    finally:
        await db.close()

    print(f"[补重要性] {'将改' if args.dry_run else '已改'} {changed} 条")
    print("[补重要性] 按类型分布：")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"   {k or '(空)'}: {v}")


asyncio.run(main())
