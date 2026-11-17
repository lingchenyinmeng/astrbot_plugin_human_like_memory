"""字段名一致性检查：把「字段名写错」的三条路径都覆盖。

1. 普通构造      Foo(bar=1)          → bar 不存在会 TypeError
2. from_dict     cls(bar=...)        → 同上
3. from_dict     d.get("bar")        → **不会报错**，静默拿到 None/默认值 ← 最难发现

第 3 种最阴：拼错一个 key，程序照跑，只是那个字段永远是默认值。
"""
import ast
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(HERE)
SIBLING = os.path.join(os.path.dirname(PLUGIN_ROOT), "astrbot_plugin_human_like_personality")
ROOTS = [PLUGIN_ROOT] + ([SIBLING] if os.path.isdir(SIBLING) else [])

# 这些不是"字段"，是嵌套 dict 里的 key，允许出现
NESTED_OK = {
    # Gestalt / Schema / Adaptations 里嵌套结构用到的 key
    "hoped_for", "feared", "tensions", "scripts", "public_self",
    "attachment_levels", "possible_selves", "self_efficacy",
    "locus_of_control", "attribution_style", "altruism",
    "boundaries", "existence_fear", "existence_anchor", "locked",
    "行为", "信念", "解释", "behavior", "belief", "resolution",
    "群里", "私下", "差异", "对他人", "对自己",
    "内部", "稳定", "全局", "主要", "次要", "边界",
    # 历史兼容字段：1.2.x 的老数据只有 attachment_style，
    # from_dict 里专门有一段把它升级成 attachment_levels。不是 bug。
    "attachment_style",
}

SKIP_NAMES = {"dict", "list", "set", "tuple", "str", "int", "float", "bool",
              "Counter", "defaultdict", "datetime", "Path"}


class Collector(ast.NodeVisitor):
    def __init__(self):
        self.classes: dict[str, set[str]] = {}
        self.bases: dict[str, list[str]] = {}
        self.plain_calls: list[tuple[str, int, set[str]]] = []
        self.cls_calls: list[tuple[str, int, set[str]]] = []
        self.get_keys: list[tuple[str, int, str]] = []
        self._cur: str | None = None
        self._in_from_dict = False

    def resolve(self, cls: str, _seen=None) -> set[str]:
        if _seen is None:
            _seen = set()
        if cls in _seen:
            return set()
        _seen.add(cls)
        out = set(self.classes.get(cls, set()))
        for b in self.bases.get(cls, []):
            out |= self.resolve(b, _seen)
        return out

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        is_dc = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            for d in node.decorator_list
        )
        if is_dc:
            self.classes[node.name] = {
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
            self.bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
        prev = self._cur
        self._cur = node.name
        self.generic_visit(node)
        self._cur = prev

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_from_dict
        self._in_from_dict = node.name in ("from_dict", "_from_dict")
        self.generic_visit(node)
        self._in_from_dict = prev

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        kwargs = {kw.arg for kw in node.keywords if kw.arg}

        if name == "cls" and self._cur and kwargs:
            self.cls_calls.append((self._cur, node.lineno, kwargs))
        elif name and name not in SKIP_NAMES and kwargs:
            self.plain_calls.append((name, node.lineno, kwargs))

        # d.get("xxx") —— 只在 from_dict 里查
        if (
            self._in_from_dict
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and self._cur
        ):
            self.get_keys.append((self._cur, node.lineno, node.args[0].value))

        self.generic_visit(node)


def main() -> None:
    problems: list[str] = []
    n_plain = n_cls = n_get = 0

    for root in ROOTS:
        if not os.path.isdir(root):
            print(f"!! 目录不存在: {root}")
            continue
        plugin = os.path.basename(root)
        print(f"\n{'=' * 66}\n插件: {plugin}\n{'=' * 66}")

        py_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
            for fn in filenames:
                if fn.endswith(".py"):
                    py_files.append(os.path.join(dirpath, fn))

        defined: dict[str, set[str]] = {}
        for f in py_files:
            try:
                tree = ast.parse(open(f, encoding="utf-8-sig").read(), filename=f)
            except (SyntaxError, UnicodeDecodeError) as e:
                problems.append(f"{plugin}: 解析失败 {os.path.relpath(f, root)}: {e}")
                continue
            c = Collector()
            c.visit(tree)
            for cls in c.classes:
                defined[cls] = c.resolve(cls)

        print(f"  dataclass {len(defined)} 个: {sorted(defined)}")

        for f in py_files:
            rel = os.path.relpath(f, root)
            try:
                tree = ast.parse(open(f, encoding="utf-8-sig").read(), filename=f)
            except (SyntaxError, UnicodeDecodeError):
                continue
            c = Collector()
            c.visit(tree)

            for name, lineno, kwargs in c.plain_calls:
                if name not in defined:
                    continue
                n_plain += 1
                extra = kwargs - defined[name]
                if extra:
                    problems.append(f"{plugin}/{rel}:{lineno}  {name}(...) 字段不存在 {sorted(extra)}")

            for cls_name, lineno, kwargs in c.cls_calls:
                if cls_name not in defined:
                    continue
                n_cls += 1
                extra = kwargs - defined[cls_name]
                if extra:
                    problems.append(
                        f"{plugin}/{rel}:{lineno}  {cls_name}.from_dict 的 cls(...) 字段不存在 {sorted(extra)}"
                    )

            for cls_name, lineno, key in c.get_keys:
                if cls_name not in defined:
                    continue
                n_get += 1
                if key in NESTED_OK:
                    continue
                if key not in defined[cls_name]:
                    problems.append(
                        f"{plugin}/{rel}:{lineno}  {cls_name}.from_dict 读了 d.get(\"{key}\")，"
                        f"但 {cls_name} 没有这个字段（静默取默认值）"
                    )

    print(f"\n{'=' * 66}")
    print(f"检查：普通构造 {n_plain} / from_dict cls {n_cls} / d.get key {n_get}")
    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  !! {p}")
        raise SystemExit(1)
    print("\n没有发现字段不匹配")


if __name__ == "__main__":
    main()
