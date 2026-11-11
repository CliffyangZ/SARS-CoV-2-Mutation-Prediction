"""Research/ 下所有公開函式／類別都要有含中文字元的 docstring（SC-010、FR-028）。"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "Research"


def test_all_public_have_chinese_docstring():
    missing = []
    for f in ROOT.rglob("*.py"):
        tree = ast.parse(f.read_text())
        if f.name != "__init__.py" and not re.search(r"[一-鿿]", ast.get_docstring(tree) or ""):
            missing.append(f"{f.name}: 模組")
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
                if not re.search(r"[一-鿿]", ast.get_docstring(n) or ""):
                    missing.append(f"{f.name}:{n.name}")
    assert not missing, missing
