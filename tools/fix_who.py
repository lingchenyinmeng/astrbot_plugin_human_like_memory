"""补 who_id 和 person_ids（铃光反馈的一、二）。

一、who_id 空 422 条：
   234 条 who_name 也空 → 救不了，跳过
   188 条有 who_name   → 名字查人库，反查 QQ 号补上

二、person_ids 空 355 条：
   234 条 who 全空      → 跳过
   34 条只有 who_name   → 名字认人
   87 条有 who_id       → 直接认

顺手：清掉 expired（已经不参与检索，赖在库里只是占地方）。

用法：
    python tools/fix_who.py --dry-run
    python tools/fix_who.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

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
    ap = argparse.ArgumentParser(description="补 who_id / person_ids")
    ap.add_argument("--memory-db", default=mdb)
    ap.add_argument("--people-db", default=pdb)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-expired", action="store_true", help="顺手清 expired")
    args = ap.parse_args()

    if not args.memory_db or not os.path.isfile(args.memory_db):
        print(f"[补who] 找不到记忆库: {args.memory_db}")
        raise SystemExit(2)
    if not args.people_db:
        print("[补who] 找不到 people.db")
        raise SystemExit(2)

    people = PeopleStore(args.people_db)
    await people.initialize()

    # 预载：名字 → person_id，person_id → QQ 号
    name2pid: dict[str, str] = {}
    pid2qq: dict[str, str] = {}
    async with people._connect() as pdb_conn:
        cur = await pdb_conn.execute(
            "SELECT person_id, kind, value FROM person_tags"
        )
        for r in await cur.fetchall():
            k, v = str(r["kind"] or ""), str(r["value"] or "").strip()
            if not v:
                continue
            if k in ("nickname", "real_name", "alias"):
                name2pid.setdefault(v, str(r["person_id"]))
            elif k == "qq":
                pid2qq.setdefault(str(r["person_id"]), v)
    print(f"[补who] 人库：{len(name2pid)} 个名字，{len(pid2qq)} 个有 QQ 号")

    db = await aiosqlite.connect(args.memory_db)
    db.row_factory = aiosqlite.Row
    try:
        # ---------- 一、补 who_id ----------
        cur = await db.execute(
            "SELECT id, who_name FROM atoms WHERE COALESCE(who_id,'')='' "
            "AND COALESCE(who_name,'')!='' ORDER BY id"
        )
        rows = await cur.fetchall()
        fixed_id = 0
        for r in rows:
            name = str(r["who_name"] or "").strip()
            pid = name2pid.get(name)
            if not pid or pid == ME_ID:
                continue
            qq = pid2qq.get(pid)
            if not qq:
                continue
            fixed_id += 1
            if not args.dry_run:
                await db.execute(
                    "UPDATE atoms SET who_id = ? WHERE id = ?",
                    (qq, int(r["id"])),
                )
        if not args.dry_run:
            await db.commit()
        print(f"[补who] who_id: 补了 {fixed_id} / {len(rows)} 条（剩下的是人库里也没 QQ 号）")

        # ---------- 二、补 person_ids ----------
        cur = await db.execute(
            "SELECT id, who_id, who_name FROM atoms "
            "WHERE person_ids = '[]' OR COALESCE(person_ids,'')='' ORDER BY id"
        )
        rows2 = await cur.fetchall()
        fixed_pid = 0
        for r in rows2:
            wid = str(r["who_id"] or "").strip()
            wname = str(r["who_name"] or "").strip()
            if not wid and not wname:
                continue
            pid = ""
            # 先按 QQ 号找
            if wid:
                pid = next((p for p, q in pid2qq.items() if q == wid), "")
            # 再按名字找
            if not pid and wname:
                pid = name2pid.get(wname, "")
            if not pid or pid == ME_ID:
                continue
            fixed_pid += 1
            if not args.dry_run:
                await db.execute(
                    "UPDATE atoms SET person_ids = ? WHERE id = ?",
                    (json.dumps([pid], ensure_ascii=False), int(r["id"])),
                )
        if not args.dry_run:
            await db.commit()
        print(f"[补who] person_ids: 补了 {fixed_pid} / {len(rows2)} 条")

        # ---------- 三、清 expired ----------
        if args.clean_expired:
            cur = await db.execute("SELECT COUNT(*) FROM atoms WHERE status='expired'")
            n = int((await cur.fetchone())[0])
            if not args.dry_run and n:
                await db.execute("DELETE FROM atoms WHERE status='expired'")
                await db.commit()
                print(f"[补who] 清掉 expired: {n} 条")
            else:
                print(f"[补who] expired: {'将清' if args.dry_run else '没有'} {n} 条")
    finally:
        await db.close()


asyncio.run(main())
