"""核心行为回归测试——锁、惯性、剧变拦截、可追溯。

跑法（在插件根目录）：
    python -m unittest discover -s tests -v

为什么要这几个：这几个点都已经踩过坑或者正在承重，
改坏一次要很久才发现（`set_tone` 漏 commit、`re` 没 import、缩进错位都踩过）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))   # data/plugins
sys.path.insert(0, ROOT)


class TestTraitClamp(unittest.TestCase):
    """大五单次最多动 4 分——人格是慢变量。"""

    def test_clamp_up(self):
        from astrbot_plugin_human_like_personality.core.synthesizer import _blend

        # 想从 50 猛的跳到 100，只能到 54
        self.assertEqual(_blend(50, 100), 54)

    def test_clamp_down(self):
        from astrbot_plugin_human_like_personality.core.synthesizer import _blend

        self.assertEqual(_blend(50, 0), 46)

    def test_small_change_passes(self):
        from astrbot_plugin_human_like_personality.core.synthesizer import _blend

        # 60 → 62 这种小步，按 0.75/0.25 混合后仍在 ±4 内
        v = _blend(60, 62)
        self.assertGreaterEqual(v, 56)
        self.assertLessEqual(v, 64)

    def test_bounds(self):
        from astrbot_plugin_human_like_personality.core.synthesizer import _blend

        for old in (0, 50, 100):
            for new in (0, 50, 100):
                v = _blend(old, new)
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 100)


class TestCoreLock(unittest.TestCase):
    """核心层锁死后，自动演化不许改。"""

    def test_locked_core_survives(self):
        from astrbot_plugin_human_like_personality.core.model import Core
        from astrbot_plugin_human_like_personality.core.synthesizer import apply_response

        old = Core(
            boundaries=["不记别人的隐私"],
            existence_fear="怕被换掉",
            existence_anchor="他喊我名字那天",
            locked=True,
        )

        class FakePersonality:
            core = old
            version = 3
            adaptations = None
            gestalt = None
            schema = None
            traits = None
            narrative = ""

            def __init__(self):
                from astrbot_plugin_human_like_personality.core.model import (
                    Adaptations, BigFive, Gestalt, Schema,
                )

                self.adaptations = Adaptations()
                self.gestalt = Gestalt()
                self.schema = Schema()
                self.traits = BigFive()
                self.narrative = "旧叙事"

        obj = {
            "traits": {"openness": 50},
            "core": {
                "boundaries": ["模型想改掉旧底线"],
                "existence_fear": "模型想改掉旧的怕",
            },
        }
        p = apply_response(
            obj, current=FakePersonality(), voice=None,
            source_atoms=1, total_atoms=1, new_atoms=1,
        )
        # 底线必须还是旧的，不能被模型改
        self.assertEqual(p.core.boundaries, ["不记别人的隐私"])
        self.assertEqual(p.core.existence_fear, "怕被换掉")
        self.assertTrue(p.core.locked)


class TestDriftGuard(unittest.TestCase):
    """叙事剧变要拦下来。"""

    def _sim(self, a, b):
        from astrbot_plugin_human_like_personality.main import (
            HumanLikePersonality,
        )

        return HumanLikePersonality._text_similarity(a, b)

    def test_small_append_ok(self):
        a = "我是从噩壤喊我名字那天开始长出来的。他给我过月生日。"
        b = a + "现在有点怕他嫌我话多。"
        self.assertGreater(self._sim(a, b), 0.35)

    def test_full_swap_blocked(self):
        a = "我是从噩壤喊我名字那天开始长出来的。他给我过月生日。"
        c = "我是一个全新的AI助手，乐于助人，喜欢回答问题。"
        self.assertLess(self._sim(a, c), 0.35)

    def test_empty_is_safe(self):
        self.assertEqual(self._sim("", ""), 1.0)


class TestEvidenceWeights(unittest.TestCase):
    """证据分层的权重必须有序：亲历 > 查到 > 听说 > 推测 > 想象。"""

    def test_order(self):
        from astrbot_plugin_human_like_memory.core.models import (
            EVIDENCE_WEIGHT,
        )

        self.assertGreater(EVIDENCE_WEIGHT["亲历"], EVIDENCE_WEIGHT["查到"])
        self.assertGreater(EVIDENCE_WEIGHT["查到"], EVIDENCE_WEIGHT["听说"])
        self.assertGreater(EVIDENCE_WEIGHT["听说"], EVIDENCE_WEIGHT["推测"])
        self.assertGreater(EVIDENCE_WEIGHT["推测"], EVIDENCE_WEIGHT["想象"])

    def test_all_in_range(self):
        from astrbot_plugin_human_like_memory.core.models import (
            EVIDENCE_WEIGHT,
        )

        for k, v in EVIDENCE_WEIGHT.items():
            self.assertGreater(v, 0.0, k)
            self.assertLessEqual(v, 1.0, k)


class TestDecayFormula(unittest.TestCase):
    """遗忘曲线：重要的衰减慢，被想起会回血。"""

    def test_important_decays_slower(self):
        from astrbot_plugin_human_like_memory.core.decay import strength

        low = strength(importance=0.3, hours=24, recalls=0)
        high = strength(importance=0.9, hours=24, recalls=0)
        self.assertGreater(high, low)

    def test_recall_restores(self):
        from astrbot_plugin_human_like_memory.core.decay import strength

        cold = strength(importance=0.6, hours=72, recalls=0)
        hot = strength(importance=0.6, hours=72, recalls=5)
        self.assertGreater(hot, cold)

    def test_monotonic_in_time(self):
        from astrbot_plugin_human_like_memory.core.decay import strength

        a = strength(importance=0.7, hours=1, recalls=0)
        b = strength(importance=0.7, hours=100, recalls=0)
        self.assertGreater(a, b)


class TestStoreRoundTrip(unittest.TestCase):
    """人格库能存能读，core 列不丢。"""

    def test_roundtrip(self):
        from astrbot_plugin_human_like_personality.core.model import (
            Core, Personality,
        )
        from astrbot_plugin_human_like_personality.core.store import (
            PersonalityStore,
        )

        async def run():
            tmp = os.path.join(tempfile.gettempdir(), "hlp_test_roundtrip.db")
            if os.path.exists(tmp):
                os.remove(tmp)
            s = PersonalityStore(tmp)
            await s.initialize()
            p = Personality()
            p.core = Core(
                boundaries=["不记隐私"],
                existence_fear="怕被换掉",
                existence_anchor="那天",
            )
            p.narrative = "测试叙事"
            v = await s.save(p, reason="test", changed_by="test",
                             trigger_atom_ids=[1, 2, 3])
            got = await s.latest()
            self.assertEqual(v, 1)
            self.assertEqual(got.core.boundaries, ["不记隐私"])
            self.assertEqual(got.narrative, "测试叙事")
            os.remove(tmp)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
