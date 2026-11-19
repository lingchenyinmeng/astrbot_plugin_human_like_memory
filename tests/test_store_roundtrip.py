"""库读写回归测试——专门防止「加了列但忘了加 dataclass 字段」这类错。

这个测试存在的原因：
`recall_count` / `valence` / `arousal` 三列加进了 SQL 和 `_row_to_atom`，
但忘了加到 `Atom` 的 dataclass 字段上，线上直接报
`Atom.__init__() got an unexpected keyword argument 'recall_count'`。

这类错单元测试看不出来——必须真的走一遍「写库 → 读库」。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)


class TestAtomRoundTrip(unittest.TestCase):
    """写进去再读出来，所有字段都要在。"""

    def test_all_fields_survive(self):
        from astrbot_plugin_human_like_memory.core.models import Atom
        from astrbot_plugin_human_like_memory.core.store import MemoryStore

        async def run():
            tmp = os.path.join(tempfile.gettempdir(), "hlm_roundtrip_test.db")
            if os.path.exists(tmp):
                os.remove(tmp)
            store = MemoryStore(tmp, {})
            await store.initialize()

            a = Atom(
                content="我翻了日志查到是显存不够",
                mtype="fact",
                nature="fact",
                layer="me_to_owner",
                who_id="10001",
                who_name="测试者",
                importance=0.8,
                status="active",
                # 这三个就是出事的那三个
                evidence="查到",
                recall_count=3,
                valence=0.25,
                arousal=0.4,
            )
            a.ttl_days = a.compute_ttl()
            await store.add_atoms([a])

            # 读回来
            rows = await store.search_atoms("显存", limit=5)
            self.assertTrue(rows, "没读到刚写进去的原子")
            got = rows[0]
            self.assertEqual(got.evidence, "查到")
            self.assertEqual(got.recall_count, 3)
            self.assertAlmostEqual(got.valence, 0.25, places=2)
            self.assertAlmostEqual(got.arousal, 0.4, places=2)
            os.remove(tmp)

        asyncio.run(run())

    def test_atom_accepts_new_kwargs(self):
        """构造函数直接收不到就会炸——这是最便宜的守门。"""
        from astrbot_plugin_human_like_memory.core.models import Atom

        a = Atom(
            content="x",
            evidence="亲历",
            recall_count=1,
            valence=-0.5,
            arousal=0.9,
        )
        self.assertEqual(a.evidence, "亲历")
        self.assertEqual(a.recall_count, 1)
        self.assertAlmostEqual(a.valence, -0.5)

    def test_touch_extends_expiry(self):
        """被想起要续命——`touch` 以前只更新 last_access。"""
        from astrbot_plugin_human_like_memory.core.models import Atom
        from astrbot_plugin_human_like_memory.core.store import MemoryStore

        async def run():
            tmp = os.path.join(tempfile.gettempdir(), "hlm_touch_test.db")
            if os.path.exists(tmp):
                os.remove(tmp)
            store = MemoryStore(tmp, {})
            await store.initialize()
            a = Atom(content="会被想起的事", importance=0.6, status="active")
            a.ttl_days = 1.0
            await store.add_atoms([a])

            # store._connect() 返回的是 aiosqlite 的 connection，不是 cursor
            # 先把 expires_at 压到 1 小时后，否则 compute_ttl() 算出来 26 天，
            # MAX(26天, 7天) = 26天，续命看不出来
            async with store._connect() as db:
                cur = await db.execute(
                    "SELECT id FROM atoms WHERE content = ?", ("会被想起的事",)
                )
                row = await cur.fetchone()
                atom_id = int(row[0])
                import time as _t

                short_exp = _t.time() + 3600.0
                await db.execute(
                    "UPDATE atoms SET expires_at = ? WHERE id = ?",
                    (short_exp, atom_id),
                )
                await db.commit()

            await store.touch(atom_id)

            async with store._connect() as db:
                cur = await db.execute(
                    "SELECT expires_at, recall_count FROM atoms WHERE id = ?",
                    (atom_id,),
                )
                row = await cur.fetchone()
            self.assertGreater(float(row[0]), short_exp, "touch 没续命")
            self.assertEqual(int(row[1]), 1, "touch 没记账")
            os.remove(tmp)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
