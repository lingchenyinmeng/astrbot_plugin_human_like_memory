"""修 self-check 报告里的 P1 / P2 / P4 / P5。

P1：96 条熟人边 tone 全空 → 补「未形成看法」占位，别让空串冒充已建边
P2：relation_evidence 0 条 → 给每条 inferred 边补一条证据（引用最高频的同框原子）
P4：pending 484 条是截断的人名碎片 → 标 resolved=2（无效）而不是假装已解决
P5：34 条 who_name 是碎片（"鲸宝随即说"）→ 按前缀匹配人库真名，匹配不上就清空

用法：
    python tools/fix_selfcheck.py --dry-run
    python tools/fix_selfcheck.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PLUGIN_ROOT))
sys.path.insert(0, PLUGIN_ROOT)

import aiosqlite

from astrbot_plugin_human_like_memory.core.people import (  # noqa: E402
    PeopleStore,
    ME_ID,
    SOURCE_INFERRED,
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
    ap = argparse.ArgumentParser(description="修自查报告 P1/P2/P4/P5")
    ap.add_argument("--memory-db", default=mdb)
    ap.add_argument("--people-db", default=pdb)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.memory_db or not os.path.isfile(args.memory_db):
        print(f"[自查修复] 找不到记忆库: {args.memory_db}")
        raise SystemExit(2)

    people = PeopleStore(args.people_db)
    await people.initialize()

    # 预载人库名字（含 QQ）
    name2pid: dict[str, str] = {}
    pid2qq: dict[str, str] = {}
    async with people._connect() as conn:
        cur = await conn.execute("SELECT person_id, kind, value FROM person_tags")
        for r in await cur.fetchall():
            k, v = str(r["kind"] or ""), str(r["value"] or "").strip()
            if not v:
                continue
            if k in ("nickname", "real_name", "alias"):
                name2pid.setdefault(v, str(r["person_id"]))
            elif k == "qq":
                pid2qq.setdefault(str(r["person_id"]), v)
    # 按长度倒序，匹配时优先长名字
    known_names = sorted(name2pid.keys(), key=len, reverse=True)

    db = await aiosqlite.connect(args.memory_db)
    db.row_factory = aiosqlite.Row
    try:
        # ---------- P5：清洗 who_name 碎片 ----------
        cur = await db.execute(
            "SELECT id, who_name FROM atoms WHERE COALESCE(who_id,'')='' "
            "AND COALESCE(who_name,'')!='' ORDER BY id"
        )
        rows = await cur.fetchall()
        fixed_name = 0
        cleared_name = 0
        for r in rows:
            raw = str(r["who_name"] or "").strip()
            if not raw:
                continue
            # 已经是干净名字就跳过
            if raw in name2pid:
                continue
            # 按前缀找最长的已知名字
            hit = ""
            for n in known_names:
                if n and raw.startswith(n):
                    hit = n
                    break
            if hit:
                pid = name2pid[hit]
                qq = pid2qq.get(pid, "")
                fixed_name += 1
                if not args.dry_run:
                    await db.execute(
                        "UPDATE atoms SET who_name = ?, who_id = ? WHERE id = ?",
                        (hit, qq, int(r["id"])),
                    )
            else:
                # 匹配不上就清空（碎片留着只会污染匹配）
                cleared_name += 1
                if not args.dry_run:
                    await db.execute(
                        "UPDATE atoms SET who_name = '' WHERE id = ?",
                        (int(r["id"]),),
                    )
        if not args.dry_run:
            await db.commit()
        print(f"[P5] who_name 碎片：修正 {fixed_name} 条，清空 {cleared_name} 条")

        # ---------- P2：给 inferred 边补证据 ----------
        async with people._connect() as conn:
            cur = await conn.execute(
                "SELECT edge_id, a_id, b_id, note FROM relations "
                "WHERE source = ? AND to_ts IS NULL",
                (SOURCE_INFERRED,),
            )
            edges = [dict(r) for r in await cur.fetchall()]
        print(f"[P2] 需要补证据的 inferred 边: {len(edges)} 条")

        # 先扫一遍原子，建 (a,b) → [原子片段]
        pair_samples: dict[tuple[str, str], list[str]] = defaultdict(list)
        cur = await db.execute(
            "SELECT content, person_ids FROM atoms "
            "WHERE layer='other_to_other' AND COALESCE(person_ids,'[]')!='[]'"
        )
        for r in await cur.fetchall():
            try:
                pids = json.loads(r["person_ids"] or "[]")
            except Exception:
                continue
            pids = sorted({str(p) for p in pids if p and str(p) != ME_ID})
            if len(pids) < 2:
                continue
            snippet = str(r["content"] or "")[:60]
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    pair_samples[(pids[i], pids[j])].append(snippet)

        ev_added = 0
        for e in edges:
            a, b = str(e["a_id"]), str(e["b_id"])
            key = tuple(sorted((a, b)))
            samples = pair_samples.get(key) or []
            if not samples:
                continue
            quote = samples[0]
            # 已存在同样证据就跳过
            async with people._connect() as conn:
                cur = await conn.execute(
                    "SELECT id FROM relation_evidence WHERE edge_id = ? LIMIT 1",
                    (int(e["edge_id"]),),
                )
                if await cur.fetchone() is not None:
                    continue
            if args.dry_run:
                ev_added += 1
                continue
            async with people._connect() as conn:
                await conn.execute(
                    "INSERT INTO relation_evidence(edge_id, a_id, b_id, rel_type, "
                    "speaker, speaker_kind, quote, weight, created_at) "
                    "VALUES (?, ?, ?, '熟人', 'inferred', 'inferred', ?, ?, ?)",
                    (int(e["edge_id"]), a, b, quote,
                     round(0.3 + 0.05 * len(samples), 2), __import__("time").time()),
                )
                await conn.commit()
            ev_added += 1
        print(f"[P2] 补证据 {ev_added} 条")

        # ---------- P1：给空 tone 的边写占位 ----------
        async with people._connect() as conn:
            cur = await conn.execute(
                "UPDATE relations SET tone = '（还没形成看法，只是认识）' "
                "WHERE source = ? AND to_ts IS NULL AND TRIM(COALESCE(tone,''))=''",
                (SOURCE_INFERRED,),
            )
            n_tone = cur.rowcount
            await conn.commit()
        print(f"[P1] 给 {n_tone} 条空壳边写了占位底色")

        # ---------- P4：pending 脏数据 ----------
        async with people._connect() as conn:
            cur = await conn.execute(
                "SELECT id, payload FROM pending WHERE resolved = 0 OR resolved = 1"
            )
            prows = await cur.fetchall()
            junk_ids = []
            for r in prows:
                try:
                    p = json.loads(r["payload"] or "{}")
                except Exception:
                    junk_ids.append(int(r["id"]))
                    continue
                qq = str(p.get("qq") or "").strip()
                name = str(p.get("name") or "").strip()
                # 没 QQ + 名字不像人名（太长/带动作词/引号没闭合）
                looks_junk = (
                    not qq
                    and (
                        not name
                        or len(name) > 8
                        or re.search(r"(说|要求|让|叫|希望|提醒|告诉|问|夸|骂|交代|随即)", name)
                        or name.count("「") != name.count("」")
                    )
                )
                if looks_junk:
                    junk_ids.append(int(r["id"]))
            if not args.dry_run and junk_ids:
                ph = ",".join("?" * len(junk_ids))
                await conn.execute(
                    f"UPDATE pending SET resolved = 2 WHERE id IN ({ph})", junk_ids
                )
                await conn.commit()
            print(f"[P4] pending 标无效: {len(junk_ids)} 条（resolved=2）")
    finally:
        await db.close()

    print("[自查修复] 完成")


asyncio.run(main())
