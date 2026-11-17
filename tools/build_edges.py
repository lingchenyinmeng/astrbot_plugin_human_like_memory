"""从 other_to_other 原子生成「别人之间」的弱关系边。

原则：person_ids 是「这条消息里出现了谁」，不是「谁和谁有关系」。
所以不直接连，而是统计**同框次数**——两个人一起出现在 ≥2 条事件里，
才认为「他们之间有来往」，生成一条低置信的「认识」边。

同框 1 次不生成（噪声太多）。

用法：
    python tools/build_edges.py --dry-run
    python tools/build_edges.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PLUGIN_ROOT))
sys.path.insert(0, PLUGIN_ROOT)

import aiosqlite

from astrbot_plugin_human_like_memory.core.people import (  # noqa: E402
    PeopleStore,
    ME_ID,
)
from astrbot_plugin_human_like_memory.tools.migrate_from_livingmemory import (  # noqa: E402
    find_data_dir,
)


def default_paths() -> tuple[str | None, str | None]:
    data = find_data_dir()
    if not data:
        return None, None
    base = os.path.join(data, "plugin_data", "astrbot_plugin_human_like_memory")
    return os.path.join(base, "memory.db"), os.path.join(base, "people.db")


async def main() -> None:
    mdb, pdb = default_paths()
    ap = argparse.ArgumentParser(description="生成别人之间的弱关系边")
    ap.add_argument("--memory-db", default=mdb)
    ap.add_argument("--people-db", default=pdb)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-cooccur", type=int, default=2, help="同框几次才建边")
    args = ap.parse_args()

    if not args.memory_db or not os.path.isfile(args.memory_db):
        print(f"[建边] 找不到记忆库: {args.memory_db}")
        raise SystemExit(2)

    # 1. 读原子，统计同框
    db = await aiosqlite.connect(args.memory_db)
    db.row_factory = aiosqlite.Row
    pairs: Counter[tuple[str, str]] = Counter()
    try:
        cur = await db.execute(
            "SELECT person_ids FROM atoms "
            "WHERE layer = 'other_to_other' AND COALESCE(person_ids,'[]') != '[]'"
        )
        n_atoms = 0
        for r in await cur.fetchall():
            try:
                pids = json.loads(r["person_ids"] or "[]")
            except Exception:
                continue
            pids = [str(p) for p in pids if p and str(p) != ME_ID]
            # 去重，避免同一条里重复计数
            uniq = sorted(set(pids))
            if len(uniq) < 2:
                continue
            n_atoms += 1
            for combo in combinations(uniq, 2):
                pairs[combo] += 1
    finally:
        await db.close()

    print(f"[建边] 带 ≥2 人的 other_to_other 原子: {n_atoms} 条")
    print(f"[建边] 出现过的配对: {len(pairs)} 组")

    # 2. 筛出达到阈值的
    qualified = [(k, v) for k, v in pairs.items() if v >= args.min_cooccur]
    qualified.sort(key=lambda kv: -kv[1])
    print(f"[建边] 同框 ≥{args.min_cooccur} 次的: {len(qualified)} 组")
    for (a, b), n in qualified[:8]:
        print(f"  {a} ↔ {b}: {n} 次")

    if args.dry_run:
        print("[建边] dry-run，没写库")
        return

    # 3. 写库
    people = PeopleStore(args.people_db)
    await people.initialize()
    made = 0
    for (a, b), n in qualified:
        conf = min(0.6, 0.3 + 0.05 * n)
        try:
            await people.add_relation(
                a, b, "熟人",
                source="inferred",
                confidence=conf,
                note=f"同框 {n} 次推断",
            )
            made += 1
        except Exception as e:
            print(f"  !! {a}↔{b} 失败: {e!r}")
    print(f"[建边] 写入 {made} 条边（rel_type=熟人, source=inferred）")

    stats = await people.stats()
    print(f"[建边] 人库现状: {stats}")


asyncio.run(main())
