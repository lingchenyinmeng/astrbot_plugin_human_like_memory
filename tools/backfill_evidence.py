"""回填 evidence / valence / arousal（老数据没有这三样）。

证据分层是 1.9.0 才加的，2844 条老记忆全是空的——
不补的话，人格归纳时所有记忆的权重都一样，分层就白做了。

用法：
    python tools/backfill_evidence.py --dry-run
    python tools/backfill_evidence.py
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
    classify_affect,
    classify_evidence,
)
from astrbot_plugin_human_like_memory.tools.migrate_from_livingmemory import (  # noqa: E402
    default_owner_id,
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
    ap = argparse.ArgumentParser(description="回填证据分层和情绪二维")
    ap.add_argument("--db", default=default_db())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--owner-id", default="")
    args = ap.parse_args()
    owner_id = str(args.owner_id or "").strip() or default_owner_id()

    if not args.db or not os.path.isfile(args.db):
        print(f"[回填] 找不到记忆库: {args.db}")
        raise SystemExit(2)

    db = await aiosqlite.connect(args.db)
    db.row_factory = aiosqlite.Row
    try:
        cur = await db.execute(
            "SELECT id, content, layer, who_id, source, nature FROM atoms "
            "WHERE COALESCE(evidence,'') = '' ORDER BY id"
        )
        rows = await cur.fetchall()
        print(f"[回填] 需要补 {len(rows)} 条")

        dist: dict[str, int] = {}
        t0 = time.time()
        done = 0
        for r in rows:
            text = str(r["content"] or "")
            ev = classify_evidence(
                text,
                layer=str(r["layer"] or ""),
                who_id=str(r["who_id"] or ""),
                owner_id=owner_id,
                source=str(r["source"] or ""),
            )
            v, a = classify_affect(str(r["nature"] or ""), text)
            dist[ev] = dist.get(ev, 0) + 1
            done += 1
            if args.dry_run:
                continue
            await db.execute(
                "UPDATE atoms SET evidence = ?, valence = ?, arousal = ? WHERE id = ?",
                (ev, v, a, int(r["id"])),
            )
            if done % 500 == 0:
                await db.commit()
                print(f"  {done}/{len(rows)} 用时 {time.time()-t0:.0f}s", flush=True)
        if not args.dry_run:
            await db.commit()
    finally:
        await db.close()

    print(f"[回填] {'将补' if args.dry_run else '已补'} {done} 条")
    print("[回填] 证据分布：")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"   {k}: {v}")


asyncio.run(main())
