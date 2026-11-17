"""从聊天历史里挖黑话。

原理（跟 fish 的思路一样）：jieba 默认词典里查不到的词，就是这个小圈子自己造的
——游戏角色、战术名词、群友外号。

筛三条：
1. 不在 jieba 词典里（已收录的词不算黑话）
2. 至少被 2 个不同的人说过（一个人反复说可能是口癖，不算群体黑话）
3. 出现 >= min_count 次

输出按群分开存：A 群的黑话喂给 B 群，模型只会乱用。

跑法（必须用 AstrBot 的 venv，系统 Python 没装 jieba）：
    A:\\dafeiyu\\uv\\tools\\astrbot\\Scripts\\python.exe tools/mine_slang.py
    ... --apply     # 写盘
    ... --dry-run   # 只看
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PLUGIN_ROOT))
sys.path.insert(0, PLUGIN_ROOT)

HIST_DB = r"A:\dafeiyu\astrbot\data\data_v4.db"
OUT_DEFAULT = r"A:\dafeiyu\astrbot\data\plugin_data\astrbot_plugin_human_like_memory\slang.json"

# 明显不是黑话的词
STOP = {
    "什么", "怎么", "为什么", "这个", "那个", "我们", "你们", "他们", "自己",
    "现在", "时候", "可以", "没有", "不是", "就是", "还是", "已经", "应该",
    "因为", "所以", "但是", "然后", "如果", "这样", "那样", "知道", "觉得",
    "东西", "事情", "问题", "地方", "一个", "一下", "一样", "一点", "一起",
    "今天", "明天", "昨天", "现在", "刚才", "等会", "马上", "真的", "确实",
    "有点", "非常", "特别", "比较", "稍微", "一定", "肯定", "估计", "可能",
    # 星期 / 时段：这些是消息时间戳带出来的，不是人说的
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "周末", "星期",
    "上午", "下午", "凌晨", "中午", "晚上", "早上", "半夜",
    # 媒体标签
    "图片", "表情", "语音", "视频", "文件", "链接", "回复", "引用", "转发",
    "消息", "群聊", "私聊", "记录", "截图", "头像", "昵称",
    # 高频通用词（jieba 词典里没有但谁都在用，不算黑话）
    "这是", "这么", "那么", "不过", "一张", "只是", "还有", "但是", "而且",
    "于是", "然后", "虽然", "毕竟", "反正", "确实", "真的", "好像", "感觉",
    "觉得", "认为", "应该", "必须", "需要", "想要", "喜欢", "讨厌", "知道",
    "看见", "听见", "说话", "告诉", "回答", "问道", "说是", "表示", "提到",
    "角色", "文字", "情绪", "内容", "形式", "方式", "方法", "情况", "结果",
}


def load_messages(db_path: str, limit: int = 20000) -> list[tuple[str, str, str, str]]:
    """返回 [(user_id=群号, sender_id, sender_name, text)]。

    注意：`platform_id` 是平台名（'大肥鱼'/'webchat'），不是群号；
    群号在 `user_id` 里。一开始按 platform_id LIKE '%GroupMessage%' 筛，
    结果 0 条——因为这张表根本没存会话类型。
    """
    con = sqlite3.connect(f"file:{db_path.replace(chr(92), chr(47))}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT user_id, sender_id, sender_name, content "
            "FROM platform_message_history "
            "WHERE user_id GLOB '[0-9]*' AND length(user_id) >= 5 "
            "ORDER BY rowid DESC LIMIT ?",
            (max(100, limit),),
        )
        out = []
        for r in cur.fetchall():
            try:
                arr = json.loads(r[3] or "[]")
            except Exception:
                continue
            if not isinstance(arr, list):
                continue
            texts = []
            for seg in arr:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    t = str((seg.get("data") or {}).get("text") or "").strip()
                    if t:
                        texts.append(t)
            if not texts:
                continue
            out.append(
                (str(r[0] or ""), str(r[1] or ""), str(r[2] or ""), " ".join(texts))
            )
        return out
    finally:
        con.close()


def mine(
    rows: list[tuple[str, str, str, str]],
    *,
    min_count: int = 4,
    min_speakers: int = 2,
) -> dict[str, list[dict]]:
    try:
        import jieba

        jieba.setLogLevel(60)
    except ImportError:
        print("!! 没有 jieba，请用 AstrBot 的 venv 跑")
        return {}

    known = set(jieba.dt.FREQ.keys())

    # per 群：词 → (次数, 说过的 sender_id 集合)
    per_group: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "who": set()})
    )

    for platform_id, sender_id, sender_name, text in rows:
        # 去掉 @、表情、链接
        clean = re.sub(r"@\S+", " ", text)
        clean = re.sub(r"\[[^\]]{1,12}\]", " ", clean)
        clean = re.sub(r"https?://\S+", " ", clean)
        clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\u4e00-\u9fff]", " ", clean)
        if not clean.strip():
            continue
        for word in jieba.lcut(clean):
            w = word.strip()
            if len(w) < 2 or len(w) > 8:
                continue
            if w in known or w in STOP:
                continue
            if not re.search(r"[\u4e00-\u9fa5]", w):
                continue
            if w.isdigit():
                continue
            slot = per_group[platform_id][w]
            slot["n"] += 1
            if sender_id:
                slot["who"].add(sender_id)

    result: dict[str, list[dict]] = {}
    for gid, words in per_group.items():
        items = []
        for w, s in words.items():
            if s["n"] < min_count:
                continue
            if len(s["who"]) < min_speakers:
                continue
            items.append({"word": w, "n": s["n"], "speakers": len(s["who"])})
        items.sort(key=lambda x: (-x["n"], -x["speakers"]))
        if items:
            result[gid] = items[:80]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="挖群黑话")
    ap.add_argument("--hist-db", default=HIST_DB)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--min-speakers", type=int, default=2)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.hist_db):
        print(f"[黑话] 找不到历史库: {args.hist_db}")
        raise SystemExit(2)

    rows = load_messages(args.hist_db, args.limit)
    print(f"[黑话] 读到 {len(rows)} 条群消息")
    if not rows:
        return

    result = mine(rows, min_count=args.min_count, min_speakers=args.min_speakers)
    total = sum(len(v) for v in result.values())
    print(f"[黑话] 挖到 {len(result)} 个群，共 {total} 个词")
    for gid, items in list(result.items())[:5]:
        print(f"\n  群 {gid[-12:]}:")
        for it in items[:10]:
            print(f"    {it['word']:<8} {it['n']:>4} 次 / {it['speakers']} 人")

    if not args.apply:
        print("\n[黑话] 没写盘（加 --apply 才写）")
        return

    payload = {
        "generated_at": time.time(),
        "min_count": args.min_count,
        "min_speakers": args.min_speakers,
        "groups": result,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[黑话] 已写入 {args.out}")


if __name__ == "__main__":
    main()
